import os, sys, torch, torchaudio
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.io import wavfile
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm
from Loss_functions import MSE, Cosine_similarity, MSEP, AC_loss, compute_H_matrix
from Dataset_class import L, J
from performance_evaluation_unfiltered import compute_pressure_with_input as cpwi
from Test_train_split import load_test_train_data



def load_data_and_model(chosen_model):
    if chosen_model == "random":
        data_random_selection = torch.load("Saved Filters/random_selection_filters.pt")
        model = data_random_selection['selected_filters']

    if chosen_model == "baseline":
        model = torch.load("Saved Filters/baseline_filters.pt")

    if chosen_model == "classification":
        model = torch.load("Saved Filters/classification_filters.pt")

    if chosen_model == "regression":
        model = torch.load("Saved Filters/regression_filters.pt")

    if chosen_model == "interpolation":
        model = torch.load("Saved Filters/interpolation_filters.pt")

    #---Load data and split into test and traning data---
    data_test, data_train = load_test_train_data()

    #X_train=data_train[0]
    #X_test=data_test[0]

    filters_test=data_test[1]
    filters_train=data_train[1]
    
    n_srcs_test=data_test[4]
    n_srcs_train=data_train[4]
    
    RIRs_test=data_test[5]
    RIRs_train=data_train[5]

    dark_zone_mics_index=[0,1,2,3,4,5,6,7,8,9,10,11]
    bright_zone_mics_index=[12]
    x_input = torch.tensor([1])
    return dark_zone_mics_index, bright_zone_mics_index, n_srcs_test, n_srcs_train, filters_test, filters_train, RIRs_test, RIRs_train, x_input, model

def load_wav_file():
    wav_path = "relaxing-guitar-loop-v5-245859.wav"
    fs_wav, wav = wavfile.read(wav_path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav = wav[5*fs_wav : 7*fs_wav]
    wav = wav / np.max(np.abs(wav))  # scale to [-1,1]
    x_input = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    x_input = torchaudio.functional.resample(x_input, orig_freq=fs_wav, new_freq=16000)

def compute_pressure_with_input(rir: torch.Tensor, filter_q: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    n_mics, n_srcs, n_rir_samples = rir.shape
    filter_len = filter_q.shape[1]
    n_input_samples = reference.shape[-1]
    # The total combined impulse response length (h_combined) is n_rir_samples + filter_len - 1
    # The final pressure length (p) is h_combined_len + n_input_samples - 1
    output_len = n_rir_samples + filter_len + n_input_samples - 2
    
    
    # Zero pad reference for convolution
    reference_padded = F.pad(reference, (0, output_len - n_input_samples), 'constant', 0)
    p = torch.zeros((n_mics, output_len), device=rir.device)

    for m in range(n_mics):
        p_m = torch.zeros(output_len, device=rir.device)
        for s in range(n_srcs):
            # Combined filter impulse response: h_combined = RIR * filter_q (via standard convolution)
            rir_m_s = rir[m, s, :].unsqueeze(0).unsqueeze(0) # [1, 1, n_rir_samples]
            q_s = filter_q[s, :].unsqueeze(0).unsqueeze(0) # [1, 1, filter_len]
            rir_m_s = rir[m, s, :].unsqueeze(0).unsqueeze(0).float()  # cast to float32
            q_s = filter_q[s, :].unsqueeze(0).unsqueeze(0).float()    # cast to float32

            # --- CRITICAL FIX: SWAP INPUT/KERNEL FOR CONV1D ---
            # Since n_rir_samples (512) < filter_len (1024), we must swap them for F.conv1d.
            # Convolution is commutative: rir * q = q * rir
            h_combined = F.conv1d(q_s, rir_m_s, padding=0).squeeze()
            
            # Convolve h_combined with input signal x (reference) using FFT
            
            # Pad h_combined to ensure final output length matches 'output_len'
            h_combined_padded = F.pad(h_combined, (0, output_len - h_combined.shape[0]), 'constant', 0)
            
            n_fft = 2**int(np.ceil(np.log2(output_len)))
            
            H = torch.fft.rfft(h_combined_padded, n=n_fft)
            X_fft = torch.fft.rfft(reference_padded, n=n_fft).squeeze(0)
            
            P_fft = H * X_fft
            p_m_s = torch.fft.irfft(P_fft, n=n_fft)[:output_len] # Back to time domain
            
            p_m += p_m_s
        p[m, :] = p_m
    
    return p

#--------------------------------------------------------------
# Performance metric functions:
# Acoustic Contrast, PESQ, NSDP, STOI, total loss, attenuation
# -------------------------------------------------------------
def acoustic_contrast(p_C, bright_zone_mics_index, dark_zone_mics_index):
    p_B=p_C[bright_zone_mics_index]
    p_D=p_C[dark_zone_mics_index]
    e_B=torch.sum(p_B**2)
    e_D=torch.sum(p_D**2)
    M_B=len(bright_zone_mics_index)
    M_D=len(dark_zone_mics_index)
    AC=(M_D / M_B) * (e_B / e_D) if e_D.item() != 0 else torch.tensor(1e10)
    return 10*torch.log10(AC)

def loss_functions(true_filter, predicted_filter, rir_test, wav_input, B_idx, D_idx):
    mse_loss = MSE(predicted_filter, true_filter)
    cosine_loss = Cosine_similarity(predicted_filter.reshape(1, L*J), true_filter.reshape(1, L*J))
    msep_loss_B, _ = MSEP(predicted_filter, true_filter, rir_test, wav_input, B_idx, D_idx)
    MSPE_loss = msep_loss_B
    H, _ = compute_H_matrix(rir_test)
    AC_los = AC_loss(predicted_filter, true_filter, H, B_idx, D_idx)
    return 1/4*(mse_loss+cosine_loss+MSPE_loss+AC_los), [mse_loss, cosine_loss, MSPE_loss, AC_los]

def attenuation(rir, raw_wav, filtered, zone):
    raw_signal = cpwi(rir, raw_wav)[zone]
    e_raw = torch.sum(raw_signal**2)
    e_filt = torch.sum(filtered**2)
    return 10 * np.log10(e_raw/e_filt)

def compute_nSDP(p_C: torch.Tensor, wav_input: torch.Tensor, bright_zone_mics_index, rir: torch.Tensor):
    p_B = p_C[bright_zone_mics_index]
    ref = wav_input.float()

    d_B_list = []
    rir_m = rir[bright_zone_mics_index] 
    max_len = ref.shape[-1] + rir_m.shape[-1] - 1
    mic_pressure = torch.zeros((1, max_len), device=rir.device)

    pad = len(rir_m[:,0].T)-len(wav_input) if len(wav_input) < len(rir_m[:,0].T) else 0
    for s in range(rir_m.shape[1]):
        conv_result = F.conv1d(ref.unsqueeze(0).unsqueeze(0), rir_m[:,s].unsqueeze(0), padding=pad)
        conv_result = conv_result.squeeze(0)
        conv_result = F.pad(conv_result, (0, max_len - conv_result.shape[-1]))
        mic_pressure += conv_result
    mic_pressure[mic_pressure==0] += 1e-12
    d_B_list.append(mic_pressure) 

    d_B_tensor = torch.stack(d_B_list) 
    min_len = min(d_B_tensor.shape[-1], p_B.shape[1])
    d_B_tensor = d_B_tensor[:, :min_len]
    p_B = p_B[:, :min_len]
    
    rms_ref = torch.sqrt(torch.mean(ref ** 2))
    rms_pB = torch.sqrt(torch.mean(p_B ** 2))
    ref = ref * (rms_pB / rms_ref)

    numerator = torch.sum((d_B_tensor - p_B) ** 2, dim=1)
    denominator = torch.sum(d_B_tensor ** 2, dim=1)
    nSDP = 10 * torch.log10(numerator / denominator)

    return torch.mean(nSDP)


#--------------------------------------------------------------
# Main evaluation functions
# -------------------------------------------------------------
def average_performance_metrics_with_filters(RIR_test, selected_filters, wav_input, bright_zone_mics_index_test, dark_zone_mics_index_test, true_filter):
    """
    Computes AC, PESQ, NSDP, and STOI for the entire test set using selected filters.
    
    Parameters:
        RIR_test: [n_samples, n_mics, n_srcs, n_rir_samples] torch.Tensor
        selected_filters: [n_samples, n_srcs, filter_len] torch.Tensor
        wav_input: [1, n_input_samples] torch.Tensor
        bright_zone_mics_index_test: list of bright-zone mic indices
        dark_zone_mics_index_test: list of dark-zone mic indices
    
    Returns:
        Average, min, max of each metric for bright and dark zones
    """
    AC_list =[]
    NSDP_B_list= []
    attenuation_list = []
    attenuation_list_bz=[]

    for i in tqdm(range(RIR_test.shape[0]), disable=not sys.stdout.isatty()):
        #print(f"\n--- Evaluating sample {i+1}/{RIR_test.shape[0]} ---")
        rirs = RIR_test[i]           # [n_mics, n_srcs, n_rir_samples]
        n_srcs = 3
        filter_len = 1024
        filters_flat = selected_filters[i].float()  # [3072]
        filters = filters_flat.reshape(n_srcs, filter_len)  # [3, 1024]
        true_filters_flat=true_filter[i].float()
        true_filters=true_filters_flat.reshape(n_srcs,filter_len)

        # Compute pressures with filters
        p_C = compute_pressure_with_input(rirs, filters, wav_input)

        # Compute metrics
        AC_i = float(acoustic_contrast(p_C, bright_zone_mics_index_test, dark_zone_mics_index_test))
        mean_NSDP_B = compute_nSDP(p_C, wav_input, bright_zone_mics_index_test, rirs)
        atten = attenuation(rirs, wav_input, p_C[dark_zone_mics_index_test], dark_zone_mics_index_test)
        atten_bz=attenuation(rirs, wav_input, p_C[bright_zone_mics_index_test], bright_zone_mics_index_test)
    
        # Append results
        AC_list.append(AC_i)
        NSDP_B_list.append(mean_NSDP_B)
        attenuation_list.append(atten)
        attenuation_list_bz.append(atten_bz)

    # Convert to numpy arrays
    AC_list = np.array(AC_list)
    NSDP_B_list = np.array(NSDP_B_list)
    attenuation_arr = np.array(attenuation_list)
    attenuation_arr_bz = np.array(attenuation_list_bz)

    # Compute statistics
    results = {
        "AC": (np.sqrt(np.var(AC_list)) ,np.mean(AC_list) ,np.min(AC_list), np.max(AC_list)),
        "NSDP_B": (np.sqrt(np.var(NSDP_B_list)) ,np.mean(NSDP_B_list), np.min(NSDP_B_list), np.max(NSDP_B_list)),
        "Attenuation_DZ": (np.sqrt(np.var(attenuation_arr)),np.mean(attenuation_arr), np.min(attenuation_arr), np.max(attenuation_arr)),
        "Attenuation_BZ": (np.sqrt(np.var(attenuation_arr_bz)),np.mean(attenuation_arr_bz), np.min(attenuation_arr_bz), np.max(attenuation_arr_bz))
    }
    print(f"AC (std, mean, min, max): {results['AC']}")
    print(f"NSDP Bright Zone (std, mean, min, max): {results['NSDP_B']}")
    print(f"Attenuation Dark Zone (std, mean, min, max):{results['Attenuation_DZ']}")
    print(f"Attenuation Bright Zone (std, mean, min, max):{results['Attenuation_BZ']}")

    return results

def plot_performance_metrics(AC, PESQ_B, PESQ_D, NSDP_B, NSDP_D, STOI_B, STOI_D, total_loss):
    """
    Creates boxplots for AC, PESQ, NSDP, and STOI for bright and dark zones.
    
    Parameters:
        AC: np.array of acoustic contrast values
        PESQ_B, PESQ_D: np.array of PESQ scores
        NSDP_B, NSDP_D: np.array of NSDP scores
        STOI_B, STOI_D: np.array of STOI scores
    """

    metrics = {
        "AC (dB)": [AC],
        "nSDP (dB)": [NSDP_B, NSDP_D],
        "Total Loss": [total_loss]
    }
    
    zone_labels = {
        "AC (dB)": ["All zones"],
        "nSDP (dB)": ["Bright", "Dark"],
        "Total Loss": ["All zones"]
    }
    
    plt.figure(figsize=(12, 10))
    
    for i, (metric_name, data) in enumerate(metrics.items(), 1):
        plt.subplot(3, 2, i)
        plt.boxplot(data, labels=zone_labels[metric_name],whis=[0,100])
        plt.title(metric_name)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def loss_function_evaluation(RIR_test, selected_filters, wav_input, bright_zone_mics_index_test, dark_zone_mics_index_test, true_filter):
    """
    Computes loss functions and attenuation for the entire test set using selected filters.
    
    Parameters:
        RIR_test: [n_samples, n_mics, n_srcs, n_rir_samples] torch.Tensor
        selected_filters: [n_samples, n_srcs, filter_len] torch.Tensor
        wav_input: [1, n_input_samples] torch.Tensor
        bright_zone_mics_index_test: list of bright-zone mic indices
        dark_zone_mics_index_test: list of dark-zone mic indices
    
    Returns:
        Average, min, max of each metric for bright and dark zones
    """
    tot_loss_list = []
    individual_losses = []

    for i in tqdm(range(RIR_test.shape[0]), disable=not sys.stdout.isatty()):
        rirs = RIR_test[i]           # [n_mics, n_srcs, n_rir_samples]
        n_srcs = 3
        filter_len = 1024
        filters_flat = selected_filters[i].float()  # [3072]
        filters = filters_flat.reshape(n_srcs, filter_len)  # [3, 1024]
        true_filters_flat=true_filter[i].float()
        true_filters=true_filters_flat.reshape(n_srcs,filter_len)

        # Compute metrics    
        total_loss_i, indiv_losses = loss_functions(true_filters, filters, rirs, wav_input, bright_zone_mics_index_test, dark_zone_mics_index_test)
        
        # Append results
        tot_loss_list.append(total_loss_i)
        individual_losses.append(indiv_losses)

    # Convert to numpy arrays
    tot_loss_list = np.array(tot_loss_list)
    individual_losses_arr = np.array(individual_losses)
    
    # Compute statistics
    results = {
        "Total Loss":(np.sqrt(np.var(tot_loss_list)).item(),np.mean(tot_loss_list).item(), np.min(tot_loss_list).item(),   np.max(tot_loss_list).item()),
        "Individual Losses" : (individual_losses_arr.mean(axis=0), individual_losses_arr.min(axis=0), individual_losses_arr.max(axis=0)),
    }

    print(f"Total loss Bright Zone (std, mean, min, max):{results['Total Loss']}")
    print(f"Individual Losses (MSE, Cosine, MSEP, AC) (mean, min, max):{results['Individual Losses']}")

    return results

'Choose between:'
"random"
"baseline"
"interpolation"
"regression"
"classification"

chosen_model = "regression"

dark_zone_mics_index, bright_zone_mics_index, n_srcs_test, n_srcs_train, filters_test, filters_train, RIRs_test, RIRs_train, x_input, model = load_data_and_model(chosen_model)

average_performance_metrics_with_filters(RIRs_test, model, x_input, bright_zone_mics_index, dark_zone_mics_index, filters_test)
loss_function_evaluation(RIRs_test, model, x_input, bright_zone_mics_index, dark_zone_mics_index, filters_test)


