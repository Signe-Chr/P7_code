import os, sys, torch, torchaudio
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.io import wavfile
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm
from Loss_functions import MSE, Cosine_similarity, MSEP, AC_loss, compute_H_matrix
from Dataset_generator_script import room_indices as ri
from Dataset_class import CustomDataset, L, J
from performance_evaluation_unfiltered import compute_pressure_with_input as cpwi


#Random filter selection
data_random_selection = torch.load("Saved Filters/random_selection_filters.pt")
filters_random = data_random_selection['selected_filters']

#Baseline filters
filters_baseline = torch.load("Saved Filters/baseline_filters.pt")

#Filters from classification MLP
filters_classification=torch.load("Saved Filters/classification_filters.pt")

#Filters from regression MLP
filters_regression=torch.load("Saved Filters/regression_filters.pt")

#Filters from interpolation MLP
filters_interpolation=torch.load("Saved Filters/interpolation_filters.pt")


#---Load data and split into test and traning data---

def load_data(data_dir):
    data_dir="Signes_data"
    full_data = os.listdir(data_dir)
    data_points = []
    train_points = []
    test_points = []
    for data in full_data:
        data_points.append(data)
        i = int(data.split("_")[1])
        if i not in ri[::4]:
            train_points.append(data)
        else:
            test_points.append(data)
            
    data_train=CustomDataset(data_dir,train_points)
    data_train_loader=DataLoader(data_train,batch_size=len(data_train), shuffle=False)
    data_test=CustomDataset(data_dir,test_points)
    data_test_loader=DataLoader(data_test,batch_size=len(data_test), shuffle=False)

    temp_var_train=[batch for batch in data_train_loader][0]
    temp_var_test=[batch for batch in data_test_loader][0]

    X_train=temp_var_train[0]
    X_test=temp_var_test[0]

    filters_train=temp_var_train[1]
    filters_test=temp_var_test[1]

    bright_zone_mics_index_train=temp_var_train[2]
    bright_zone_mics_index_test=temp_var_test[2]

    dark_zone_mics_index_train=temp_var_train[3]
    dark_zone_mics_index_test=temp_var_test[3]

    n_srcs_train=temp_var_train[4]
    n_srcs_test=temp_var_test[4]

    RIRs_train=temp_var_train[5]
    RIRs_test=temp_var_test[5]

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
    return (X_train, X_test, filters_train, filters_test, bright_zone_mics_index,
            bright_zone_mics_index, dark_zone_mics_index, dark_zone_mics_index, n_srcs_train, n_srcs_test,
            RIRs_train, RIRs_test, x_input)

(X_train, X_test, filters_train, filters_test, bright_zone_mics_index, bright_zone_mics_index_test,
 dark_zone_mics_index, dark_zone_mics_index_test, n_srcs_train, n_srcs_test,
 RIRs_train, RIRs_test, x_input) = load_data("Signes_data")

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
def acoustic_contrast(p_C,bright_zone_mics_index,dark_zone_mics_index):
    p_B=p_C[bright_zone_mics_index]
    p_D=p_C[dark_zone_mics_index]
    e_B=torch.sum(p_B**2)
    e_D=torch.sum(p_D**2)
    M_B=len(bright_zone_mics_index)
    M_D=len(dark_zone_mics_index)
    AC=(M_D / M_B) * (e_B / e_D) if e_D.item() != 0 else torch.tensor(1e10)
    return 10*torch.log10(AC)

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
        return np.mean(stoi_list)

    mean_B = compute_zone_stoi(bright_zone_mics_index)
    mean_D = compute_zone_stoi(dark_zone_mics_index)

    return mean_B,mean_D

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
    e_filt = torch.sum(filtered[zone]**2)
    return 10 * np.log10(e_raw/e_filt)


#--------------------------------------------------------------
# Main evaluation functions
# -------------------------------------------------------------
def average_performance_metrics_with_filters(RIR_test, selected_filters, wav_input, bright_zone_mics_index_test, dark_zone_mics_index_test,true_filter):
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
    AC_list, pesq_B_list, pesq_D_list = [], [], []
    NSDP_B_list, NSDP_D_list = [], []
    STOI_B_list, STOI_D_list = [], []
    tot_loss_list = []

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
        mean_pesq_B, mean_pesq_D = compute_pesq_unfiltered(p_C, wav_input, bright_zone_mics_index_test, dark_zone_mics_index_test)
        mean_NSDP_B, mean_NSDP_D = compute_nSDP(p_C, wav_input, bright_zone_mics_index_test, dark_zone_mics_index_test)
        mean_STOI_B, mean_STOI_D = compute_STOI(p_C, wav_input, bright_zone_mics_index_test, dark_zone_mics_index_test)
    
        # Append results
        AC_list.append(AC_i)
        pesq_B_list.append(mean_pesq_B)
        pesq_D_list.append(mean_pesq_D)
        NSDP_B_list.append(mean_NSDP_B)
        NSDP_D_list.append(mean_NSDP_D)
        STOI_B_list.append(mean_STOI_B)
        STOI_D_list.append(mean_STOI_D)

    # Convert to numpy arrays
    AC_list = np.array(AC_list)
    pesq_B_list = np.array(pesq_B_list)
    pesq_D_list = np.array(pesq_D_list)
    NSDP_B_list = np.array(NSDP_B_list)
    NSDP_D_list = np.array(NSDP_D_list)
    STOI_B_list = np.array(STOI_B_list)
    STOI_D_list = np.array(STOI_D_list)

    # Compute statistics
    results = {
        "AC": (np.sqrt(np.var(AC_list)) ,np.mean(AC_list) ,np.min(AC_list), np.max(AC_list)),
        "PESQ_B": (np.sqrt(np.var(pesq_B_list)) ,np.mean(pesq_B_list), np.min(pesq_B_list), np.max(pesq_B_list)),
        "PESQ_D": (np.sqrt(np.var(pesq_D_list)) ,np.mean(pesq_D_list), np.min(pesq_D_list), np.max(pesq_D_list)),
        "NSDP_B": (np.sqrt(np.var(NSDP_B_list)) ,np.mean(NSDP_B_list), np.min(NSDP_B_list), np.max(NSDP_B_list)),
        "NSDP_D": (np.sqrt(np.var(NSDP_D_list)) ,np.mean(NSDP_D_list), np.min(NSDP_D_list), np.max(NSDP_D_list)),
        "STOI_B": (np.sqrt(np.var(STOI_B_list)) ,np.mean(STOI_B_list), np.min(STOI_B_list), np.max(STOI_B_list)),
        "STOI_D": (np.sqrt(np.var(STOI_D_list)) ,np.mean(STOI_D_list), np.min(STOI_D_list), np.max(STOI_D_list)),
    }
    print(f"AC (std, mean, min, max): {results['AC']}")
    print(f"PESQ Bright Zone (std, mean, min, max): {results['PESQ_B']}")
    print(f"PESQ Dark Zone (std, mean, min, max): {results['PESQ_D']}")
    print(f"NSDP Bright Zone (std, mean, min, max): {results['NSDP_B']}")
    print(f"NSDP Dark Zone (std, mean, min, max): {results['NSDP_D']}")
    print(f"STOI Bright Zone (std, mean, min, max): {results['STOI_B']}")
    print(f"STOI Dark Zone (std, mean, min, max): {results['STOI_D']}")
    # Optional: plot metrics
    #plot_performance_metrics(AC_list, pesq_B_list, pesq_D_list, NSDP_B_list, NSDP_D_list, STOI_B_list, STOI_D_list, tot_loss_list)

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
        "PESQ": [PESQ_B, PESQ_D],
        "nSDP (dB)": [NSDP_B, NSDP_D],
        "STOI": [STOI_B, STOI_D],
        "Total Loss": [total_loss]
    }
    
    zone_labels = {
        "AC (dB)": ["All zones"],
        "PESQ": ["Bright", "Dark"],
        "nSDP (dB)": ["Bright", "Dark"],
        "STOI": ["Bright", "Dark"],
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
    attenuation_list = []
    individual_losses = []

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
        total_loss_i, indiv_losses = loss_functions(true_filters, filters, rirs, wav_input, bright_zone_mics_index_test, dark_zone_mics_index_test)
        atten = attenuation(rirs, wav_input, p_C[dark_zone_mics_index_test], dark_zone_mics_index_test)

        # Append results
        tot_loss_list.append(total_loss_i)
        individual_losses.append(indiv_losses)
        attenuation_list.append(atten)

    # Convert to numpy arrays
    tot_loss_list = np.array(tot_loss_list)
    individual_losses_arr = np.array(individual_losses)
    attenuation_arr = np.array(attenuation_list)

    # Compute statistics
    results = {
        "Total Loss":(np.mean(tot_loss_list), np.min(tot_loss_list),   np.max(tot_loss_list)),
        "Individual Losses" : (individual_losses_arr.mean(axis=0), individual_losses_arr.min(axis=0), individual_losses_arr.max(axis=0)),
        "Attenuation": (np.sqrt(np.var(attenuation_arr)),np.mean(attenuation_arr), np.min(attenuation_arr), np.max(attenuation_arr))
    }

    print(f"Total loss Bright Zone (mean, min, max):{results['Total Loss']}")
    print(f"Attenuation Dark Zone (std, mean, min, max):{results['Attenuation']}")
    print(f"Individual Losses (MSE, Cosine, MSEP, AC) (mean, min, max):{results['Individual Losses']}")

    return results

#filters_random, filters_baseline, filters_classification, filters_regression, filters_interpolation, filters_test

average_performance_metrics_with_filters(RIRs_test, filters_baseline, x_input, bright_zone_mics_index, dark_zone_mics_index, filters_test)
#loss_functions(RIRs_test, filters_classification, x_input, bright_zone_mics_index, dark_zone_mics_index, filters_test)

