import os,sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import torch
import torch.nn.functional as F
import numpy as np
from scipy.io import wavfile
from pesq import pesq
import torchaudio
from Test_train_split import load_test_train_data



#---Load data and split into test and traning data---
#---Load data and split into test and traning data---
data_test, data_train = load_test_train_data()

X_train=data_train[0]
X_test=data_test[0]

filters_train=data_train[1]
filters_test=data_test[1]

bright_zone_mics_index_train=data_train[2]
bright_zone_mics_index_test=data_test[2]

dark_zone_mics_index_train=data_train[3]
dark_zone_mics_index_test=data_test[3]

n_srcs_train=data_train[4]
n_srcs_test=data_test[4]

RIRs_train=data_train[5]
RIRs_test=data_test[5]

dark_zone_mics_index=[0,1,2,3,4,5,6,7,8,9,10,11]
bright_zone_mics_index=[12]

#---Load Wav file---
wav_path = "relaxing-guitar-loop-v5-245859.wav"
fs_wav, wav = wavfile.read(wav_path)
if wav.ndim > 1:
    wav = np.mean(wav, axis=1)
wav = wav[5*fs_wav : 7*fs_wav]
wav = wav / np.max(np.abs(wav))  # scale to [-1,1]
x_input = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
x_input = torchaudio.functional.resample(x_input, orig_freq=fs_wav, new_freq=16000)

#x_input = x_input.to(device)

#---Compute Acoustic contrast---
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

#---Compute Acoustic Contrast---
def acoustic_contrast(p_C,bright_zone_mics_index,dark_zone_mics_index):
    p_B=p_C[bright_zone_mics_index]
    p_D=p_C[dark_zone_mics_index]
    e_B=torch.sum(p_B**2)
    e_D=torch.sum(p_D**2)
    M_B=len(bright_zone_mics_index)
    M_D=len(dark_zone_mics_index)
    AC=(M_D / M_B) * (e_B / e_D) if e_D.item() != 0 else torch.tensor(1e10)
    return 10*torch.log10(AC)

#---Compute PESQ---
def compute_pesq_unfiltered(
    p_C,
    wav_input: torch.Tensor,
    bright_zone_mics_index_test: list[int],
    dark_zone_mics_index_test: list[int],
    fs: int=16000 ,
):

    p_B = p_C[bright_zone_mics_index_test]  # [n_bz_mics, n_samples]
    p_D = p_C[dark_zone_mics_index_test]    # [n_dz_mics, n_samples]

    # --- Step 3: Reference signal as NumPy ---
    ref = wav_input.squeeze().detach().cpu().numpy().astype(np.float32)

    # --- Step 4: Compute PESQ for Bright Zone ---
    pesq_B_scores = []
    for m in range(p_B.shape[0]):
        deg = p_B[m, :].detach().cpu().numpy().astype(np.float32)
        try:
            score = pesq(fs, ref, deg, 'wb' if fs == 16000 else 'nb')
        except Exception as e:
            print(f"[Warning] PESQ failed for BZ mic {m}: {e}")
            score = np.nan
        pesq_B_scores.append(score)

    # --- Step 5: Compute PESQ for Dark Zone ---
    pesq_D_scores = []
    for m in range(p_D.shape[0]):
        deg = p_D[m, :].detach().cpu().numpy().astype(np.float32)
        try:
            score = pesq(fs, ref, deg, 'wb' if fs == 16000 else 'nb')
        except Exception as e:
            print(f"[Warning] PESQ failed for DZ mic {m}: {e}")
            score = np.nan
        pesq_D_scores.append(score)

    # --- Step 6: Compute statistics ---
    pesq_B_scores = np.array(pesq_B_scores)
    pesq_D_scores = np.array(pesq_D_scores)

    mean_pesq_B = float(np.nanmean(pesq_B_scores))

    mean_pesq_D = float(np.nanmean(pesq_D_scores))

    return mean_pesq_B,mean_pesq_D


#---Compute NSDR---
import torch
import numpy as np

def compute_nSDP(p_C: torch.Tensor, wav_input: torch.Tensor,
                 bright_zone_mics_index: list[int],
                 dark_zone_mics_index: list[int]):
    """
    Computes normalized Signal Distortion Power (NSDP) for bright and dark zones.
    Bright zone compares against target signal.
    Dark zone compares against zero (silence target).
    """
    ref = wav_input.squeeze().detach().cpu().numpy().astype(np.float32)
    
    def compute_zone_nSDP(mic_indices, target_type="bright"):
        NSDP_list = []
        for m in mic_indices:
            deg = p_C[m, :].detach().cpu().numpy().astype(np.float32)
            min_len = min(len(ref), len(deg))
            s_est = deg[:min_len]
            
            if target_type == "bright":
                s_target = ref[:min_len]
            else:  # dark zone -> silence target
                s_target = np.zeros_like(s_est)
            
            numerator = np.sum((s_target - s_est)**2)
            denominator = np.sum(s_target**2) if target_type == "bright" else np.sum(s_est**2)
            
            # For dark zone, compare distortion power to its own signal power
            if denominator == 0:
                NSDP_val = 1e10
            else:
                NSDP_val = 10 * np.log10(numerator / denominator)
            NSDP_list.append(NSDP_val)
        return np.mean(NSDP_list)
    
    mean_B = compute_zone_nSDP(bright_zone_mics_index, "bright")
    mean_D = compute_zone_nSDP(dark_zone_mics_index, "dark")
    
    return mean_B, mean_D


#---Compute STOI---

from pystoi import stoi

def compute_STOI(p_C: torch.Tensor, wav_input: torch.Tensor,
                             bright_zone_mics_index: list[int],
                             dark_zone_mics_index: list[int],
                             fs: int = 16000):
    ref = wav_input.squeeze().detach().cpu().numpy().astype(np.float32)

    def compute_zone_stoi(mic_indices):
        stoi_list = []
        for m in mic_indices:
            deg = p_C[m, :].detach().cpu().numpy().astype(np.float32)
            min_len = min(len(ref), len(deg))
            s_target = ref[:min_len]
            s_est = deg[:min_len]
            score = stoi(s_target, s_est, fs, extended=False)
            stoi_list.append(score)
        stoi_list = np.array(stoi_list)

        return stoi_list

    b = compute_zone_stoi(bright_zone_mics_index)
    d = compute_zone_stoi(dark_zone_mics_index)

    mean_B = np.mean(b)
    mean_D = np.mean(d)



    return mean_B,mean_D

import matplotlib.pyplot as plt
import numpy as np

def plot_performance_metrics(AC, PESQ_B, PESQ_D, NSDR_B, NSDR_D, STOI_B, STOI_D):
    """
    Creates boxplots for AC, PESQ, nSDR, and STOI for bright and dark zones.
    
    Parameters:
        AC: np.array of acoustic contrast values
        PESQ_B, PESQ_D: np.array of PESQ scores
        NSDR_B, NSDR_D: np.array of nSDR scores
        STOI_B, STOI_D: np.array of STOI scores
    """

    metrics = {
        "AC (dB)": [AC],
        "PESQ": [PESQ_B, PESQ_D],
        "nSDR (dB)": [NSDR_B, NSDR_D],
        "STOI": [STOI_B, STOI_D]
    }
    
    zone_labels = {
        "AC (dB)": ["All zones"],
        "PESQ": ["Bright", "Dark"],
        "nSDR (dB)": ["Bright", "Dark"],
        "STOI": ["Bright", "Dark"]
    }
    
    plt.figure(figsize=(12, 10))
    
    for i, (metric_name, data) in enumerate(metrics.items(), 1):
        plt.subplot(2, 2, i)
        plt.boxplot(data, labels=zone_labels[metric_name])
        plt.title(metric_name)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
from performance_evaluation_unfiltered import compute_pressure_with_input as cpwi
def attenuation(rir, raw_wav, filtered,zone):
    raw_signal = cpwi(rir, raw_wav)[zone]
    e_raw = torch.sum(raw_signal**2)
    e_filt = torch.sum(filtered**2)
    return 10 * np.log10(e_raw/e_filt)

#---Compute average performance metrics across testset---
def average_performance_metrics(RIR_test, wav_input, bright_zone_mics_index_test, dark_zone_mics_index_test):
    pesq_B = []
    pesq_D = []
    AC = []
    NSDR_B = []
    NSDR_D = []
    STOI_B = []
    STOI_D = []
    attenuation_list = []
    for i in range(RIR_test.shape[0]):
        print(f"\n--- Evaluating sample {i+1}/{RIR_test.shape[0]} ---")
        rirs = RIR_test[i]
        BZ_idx = bright_zone_mics_index_test
        DZ_idx = dark_zone_mics_index_test

        # Compute pressures
        p_C = compute_pressure_with_input(rirs, wav_input)

        # Compute PESQ
        mean_pesq_B, mean_pesq_D = compute_pesq_unfiltered(p_C, wav_input, BZ_idx, DZ_idx)

        # Compute AC
        AC_i = float(acoustic_contrast(p_C, BZ_idx, DZ_idx))
        
        # Compute NSDR
        mean_NSDR_B,mean_NSDR_D=compute_nSDP(p_C,wav_input,BZ_idx,DZ_idx)
        
        #Compute STOI
        mean_STOI_B,mean_STOI_D=compute_STOI(p_C, wav_input, BZ_idx, DZ_idx)
        
        #Compute attentuation in dark zone
        atten = attenuation(rirs, wav_input, p_C[dark_zone_mics_index_test],dark_zone_mics_index_test)
        

        pesq_B.append(mean_pesq_B)
        pesq_D.append(mean_pesq_D)
        AC.append(AC_i)
        NSDR_B.append(mean_NSDR_B)
        NSDR_D.append(mean_NSDR_D)
        STOI_B.append(mean_STOI_B)
        STOI_D.append(mean_STOI_D)
        attenuation_list.append(atten)


    

    AC = np.array(AC)
    pesq_B = np.array(pesq_B)
    pesq_D = np.array(pesq_D)
    NSDR_B = np.array(NSDR_B)
    NSDR_D = np.array(NSDR_D)
    STOI_B = np.array(STOI_B)
    STOI_D = np.array(STOI_D)
    attenuation_arr = np.array(attenuation_list)

    print(f"std stoi (bz) {np.sqrt(np.var(STOI_B))}")
    print(f"std stoi (dz) {np.sqrt(np.var(STOI_D))}")

    print(f"pesq std (bz) {np.sqrt(np.var(pesq_B))}" )
    print(f"pesq std (dz) {np.sqrt(np.var(pesq_D))}" )

    avg_ac = np.mean(AC)
    print(f"std ac {np.sqrt(np.var(AC))}")
    min_ac = np.min(AC)
    max_ac = np.max(AC)

    avg_pesq_B = np.mean(pesq_B)
    min_pesq_B = np.min(pesq_B)
    max_pesq_B = np.max(pesq_B)

    avg_pesq_D = np.mean(pesq_D)
    min_pesq_D = np.min(pesq_D)
    max_pesq_D = np.max(pesq_D)
    
    avg_NSDR_B = np.mean(NSDR_B)
    min_NSDR_B = np.min(NSDR_B)
    max_NSDR_B = np.max(NSDR_B)
    
    avg_NSDR_D = np.mean(NSDR_D)
    min_NSDR_D = np.min(NSDR_D)
    max_NSDR_D = np.max(NSDR_D)
    
    avg_STOI_B = np.mean(STOI_B)
    min_STOI_B = np.min(STOI_B)
    max_STOI_B = np.max(STOI_B)
    
    avg_STOI_D = np.mean(STOI_D)
    min_STOI_D = np.min(STOI_D)
    max_STOI_D = np.max(STOI_D)
    
    avg_atten=np.mean(attenuation_arr)
    
    #plot_performance_metrics(AC, pesq_B, pesq_D, NSDR_B, NSDR_D, STOI_B, STOI_D)

    return avg_ac, min_ac, max_ac, avg_pesq_B, min_pesq_B, max_pesq_B, avg_pesq_D, min_pesq_D, max_pesq_D, avg_NSDR_B, min_NSDR_B, max_NSDR_B, avg_NSDR_D, min_NSDR_D, max_NSDR_D, avg_STOI_B, min_STOI_B, max_STOI_B, avg_STOI_D, min_STOI_D, max_STOI_D,avg_atten

if __name__=='__main__':
    avg_ac, min_ac, max_ac, avg_pesq_B, min_pesq_B, max_pesq_B, avg_pesq_D, min_pesq_D, max_pesq_D, avg_NSDR_B, min_NSDR_B, max_NSDR_B, avg_NSDR_D, min_NSDR_D, max_NSDR_D, avg_STOI_B, min_STOI_B, max_STOI_B, avg_STOI_D, min_STOI_D, max_STOI_D,avg_atten=average_performance_metrics(RIRs_test,x_input,bright_zone_mics_index,dark_zone_mics_index)
    print(f'Average AC over {RIRs_test.shape[0]} data points:{avg_ac}, minimum AC: {min_ac}, maximum AC: {max_ac}')
    print('Bright Zone:')
    print(f'Average PESQ over {RIRs_test.shape[0]} data points: {avg_pesq_B}, minimum PESQ:{min_pesq_B}, maximum pesq:{max_pesq_B}')
    print(f'Average NSDR over {RIRs_test.shape[0]} data points: {avg_NSDR_B}, minimum NSDR:{min_NSDR_B}, maximum NSDR:{max_NSDR_B}')
    print(f'Average STOI over {RIRs_test.shape[0]} data points: {avg_STOI_B}, minimum STOI:{min_STOI_B}, maximum STOI:{max_STOI_B}')
    print('Dark Zone:')
    print(f'Average PESQ over {RIRs_test.shape[0]} data points: {avg_pesq_D}, minimum PESQ:{min_pesq_D}, maximum pesq:{max_pesq_D}')
    print(f'Average NSDR over {RIRs_test.shape[0]} data points: {avg_NSDR_D}, minimum NSDR:{min_NSDR_D}, maximum NSDR:{max_NSDR_D}')
    print(f'Average STOI over {RIRs_test.shape[0]} data points: {avg_STOI_D}, minimum STOI:{min_STOI_D}, maximum STOI:{max_STOI_D}')
    print(f'Average attenuation over {RIRs_test.shape[0]} data points: {avg_atten}')