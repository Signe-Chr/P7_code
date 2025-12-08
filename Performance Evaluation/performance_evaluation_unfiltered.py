import os, sys, torch, torchaudio
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from scipy.io import wavfile
from pesq import pesq
from Test_train_split import load_test_train_data, load_wav_file, L, J, x_input_kronecker, indeces_bright, indeces_dark
from pystoi import stoi
from tqdm import tqdm
from Loss_functions import MSE, Cosine_similarity, MSEP, AC_loss, compute_H_matrix



#---Load data test and train data---
def load_data():
    data_test, data_train, data_val = load_test_train_data()
    RIRs_train=data_train[5]
    RIRs_test=data_test[5]

    return RIRs_test, RIRs_train,

#---Load Wav file---
def load_wav_file():
    wav_path = "relaxing-guitar-loop-v5-245859.wav"
    fs_wav, wav = wavfile.read(wav_path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav = wav[5*fs_wav : 7*fs_wav]
    wav = wav / np.max(np.abs(wav))  # scale to [-1,1]
    x_input = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    x_input = torchaudio.functional.resample(x_input, orig_freq=fs_wav, new_freq=16000)
    return x_input

#---Compute pressure--
def compute_pressure_with_input(rir: torch.Tensor, x_input: torch.Tensor) -> torch.Tensor:
    n_mics, n_srcs, n_rir_samples = rir.shape
    n_input_samples = x_input.shape[-1]
    
    # Output length = n_rir_samples + n_input_samples - 1
    output_len = n_rir_samples + n_input_samples - 1
    
    # Zero pad input
    x_input_padded = F.pad(x_input, (0, output_len - n_input_samples), 'constant', 0)
    p = torch.zeros((n_mics, output_len), device=rir.device)

    # FFT length (power of 2 for efficiency)
    n_fft = 2 ** int(np.ceil(np.log2(output_len)))
    X_fft = torch.fft.rfft(x_input_padded, n=n_fft).squeeze(0)

    # Loop through microphones and sources
    for m in range(n_mics):
        p_m = torch.zeros(output_len, device=rir.device)
        for s in range(n_srcs):
            h = rir[m, s, :]  # [n_rir_samples]
            h_padded = F.pad(h, (0, output_len - n_rir_samples), 'constant', 0)
            H_fft = torch.fft.rfft(h_padded, n=n_fft)
            
            # Convolution via multiplication in frequency domain
            P_fft = H_fft * X_fft
            p_m_s = torch.fft.irfft(P_fft, n=n_fft)[:output_len]
            p_m += p_m_s

        p[m, :] = p_m

    return p


#---Metrics--
def acoustic_contrast(p_C,indeces_bright, indeces_dark):
    p_B=p_C[indeces_bright]
    p_D=p_C[indeces_dark]
    e_B=torch.sum(p_B**2)
    e_D=torch.sum(p_D**2)
    M_B=len(indeces_bright)
    M_D=len(indeces_dark)
    AC=(M_D / M_B) * (e_B / e_D) if e_D.item() != 0 else torch.tensor(1e10)
    return AC

def compute_nSDP(p_C: torch.Tensor, wav_input: torch.Tensor, indeces_bright, rir: torch.Tensor):
    p_B = p_C[indeces_bright]
    ref = wav_input.float()
    #ref = F.pad(wav_input.float(), (3, 0))
    
    d_B_list = []
    rir_m = rir[indeces_bright] 
    max_len = ref.shape[-1] + rir_m.shape[-1] - 1
    mic_pressure = torch.zeros((1, max_len), device=rir.device)
    pad = len(rir_m[:,0].T)-1
    for s in range(rir_m.shape[1]):
        conv_result = F.conv1d(ref.unsqueeze(0).unsqueeze(0), rir_m[:,s].unsqueeze(0), padding=pad)
        conv_result = conv_result.squeeze(0)
        conv_result = F.pad(conv_result, (0, max_len - conv_result.shape[-1]))
        mic_pressure += conv_result
    mic_pressure[mic_pressure==0] += 1e-12
    d_B_list.append(mic_pressure)

    d_B_tensor = torch.stack(d_B_list).squeeze(0)
    min_len = min(d_B_tensor.shape[-1], p_B.shape[1])
    d_B_tensor = d_B_tensor[:, :min_len]
    p_B = p_B[:, :min_len]
    
    rms_d_B_tensor = torch.sqrt(torch.mean(d_B_tensor ** 2))
    rms_pB = torch.sqrt(torch.mean(p_B ** 2))
    d_B_tensor = d_B_tensor * (rms_pB / rms_d_B_tensor)

    numerator = torch.sum((d_B_tensor - p_B) ** 2)
    denominator = torch.sum(d_B_tensor ** 2)

    return numerator / denominator
 
def attenuation(rir, raw_wav, filtered,zone):
    raw_signal = compute_pressure_with_input(rir, raw_wav)[zone]
    e_raw = torch.sum(raw_signal**2)
    e_filt = torch.sum(filtered**2)
    return e_raw/e_filt


#---Compute average performance metrics across testset---
def average_performance_metrics(RIR_test, wav_input, indeces_bright, indeces_dark):
    AC_list = []
    NSDP_B_list= []
    attenuation_list = []
    attenuation_list_bz = []
    for i in tqdm(range(RIR_test.shape[0])):
        #print(f"\n--- Evaluating sample {i+1}/{RIR_test.shape[0]} ---")
        rirs = RIR_test[i]
        BZ_idx = indeces_bright
        DZ_idx = indeces_dark

        # Compute pressures
        p_C = compute_pressure_with_input(rirs, wav_input)

        # Compute AC
        AC_i = float(acoustic_contrast(p_C, BZ_idx, DZ_idx))
        
        # Compute NSDR
        mean_NSDP_B = compute_nSDP(p_C, wav_input, indeces_bright, rirs)
        
        #Compute attentuation in dark zone
        atten = attenuation(rirs, wav_input, p_C[indeces_dark], indeces_dark)
        atten_bz=attenuation(rirs, wav_input, p_C[indeces_bright], indeces_bright)
        
        
        AC_list.append(AC_i)
        NSDP_B_list.append(mean_NSDP_B)
        attenuation_list.append(atten)
        attenuation_list_bz.append(atten_bz)

    AC_list = np.array(AC_list)
    attenuation_arr = np.array(attenuation_list)
    attenuation_arr_bz = np.array(attenuation_list_bz)

    # Compute statistics
    results = {
        "AC": (np.sqrt(np.var(10*np.log10(AC_list))) ,10*np.log10(np.mean(AC_list)) ,np.min(10*np.log10(AC_list)), np.max(10*np.log10(AC_list))),
        "NSDP_B": (np.sqrt(np.var(10*np.log10(NSDP_B_list))) ,10*np.log10(np.mean(NSDP_B_list)), np.min(10*np.log10(NSDP_B_list)), np.max(10*np.log10(NSDP_B_list))),
        "Attenuation_DZ": (np.sqrt(np.var(10*np.log10(attenuation_arr))),10*np.log10(np.mean(attenuation_arr)), np.min(10*np.log10(attenuation_arr)), np.max(10*np.log10(attenuation_arr))),
        "Attenuation_BZ": (np.sqrt(np.var(10*np.log10(attenuation_arr_bz))),10*np.log10(np.mean(attenuation_arr_bz)), np.min(10*np.log10(attenuation_arr_bz)), np.max(10*np.log10(attenuation_arr_bz)))
    }
    print(f"AC (std, mean, min, max): {results['AC']}")
    print(f"NSDP Bright Zone (std, mean, min, max): {results['NSDP_B']}")
    print(f"Attenuation Dark Zone (std, mean, min, max):{results['Attenuation_DZ']}")
    print(f"Attenuation Bright Zone (std, mean, min, max):{results['Attenuation_BZ']}")

    return results


if __name__=='__main__':
    x_input = x_input_kronecker #load_wav_file()
    RIRs_test, RIRs_train, = load_data()
    results = average_performance_metrics(RIRs_test, x_input, indeces_bright, indeces_dark)
    
    gemt = [f"AC (std, mean, min, max): {results['AC']}\n",
                f"NSDP Bright Zone (std, mean, min, max): {results['NSDP_B']}\n", 
                f"Attenuation Dark Zone (std, mean, min, max):{results['Attenuation_DZ']}\n",
                f"Attenuation Bright Zone (std, mean, min, max):{results['Attenuation_BZ']}\n"]

    with open(f"Performance Evaluation/Results/unfiltered.txt", "w") as file:
        for string in gemt:
            file.writelines(string)