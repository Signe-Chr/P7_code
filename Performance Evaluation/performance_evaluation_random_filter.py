import os,sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import torch
import torch.nn.functional as F
import numpy as np
from Dataset_class import CustomDataset, L, J
from torch.utils.data import DataLoader
from Dataset_generator_script import room_indices as ri
from scipy.io import wavfile
from pesq import pesq
import torchaudio
from Loss_functions import compute_H_matrix, AC_tilde,w_ac,C_i
import Dataset_generator_script as dgs

data_random_selection = torch.load("random_selection_data.pt")
selected_filters = data_random_selection['selected_filters']



#---Load data and split into test and traning data---
data_dir="Signes_data"
full_data = os.listdir(data_dir)
data_points = []
train_points = []
test_points = []
for data in full_data:
    i = int(data.split("_")[1])
    if (i in ri) and (i not in ri[::4]):
        train_points.append(data)
        data_points.append(data)
    else:
        test_points.append(data)
        data_points.append(data)
        
data_train=CustomDataset(data_dir,train_points)
data_train_loader=DataLoader(data_train,batch_size=len(data_train), shuffle=True)
data_test=CustomDataset(data_dir,test_points)
data_test_loader=DataLoader(data_test,batch_size=len(data_test), shuffle=True)

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
#---Compute Cosine Similairty---
def compute_cosine_similarity(true_filter, predicted_filter):
    true_flat = true_filter.flatten()
    pred_flat = predicted_filter.flatten()
    dot_product = np.dot(true_flat, pred_flat)
    norm_true = np.linalg.norm(true_flat)
    norm_pred = np.linalg.norm(pred_flat)
    cosine_similarity = dot_product / (norm_true * norm_pred + 1e-12)
    return cosine_similarity


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
    Computes normalized Signal-to-Distortion Ratio (nSDR) for bright and dark zones.

    Parameters:
        p_C: [n_mics, n_samples] tensor of simulated pressures
        wav_input: [1, n_samples] reference signal (torch tensor)
        bright_zone_mics_index: list of bright-zone mic indices
        dark_zone_mics_index: list of dark-zone mic indices

    Returns:
        Dictionary with mean, min, max nSDR for bright and dark zones
    """
    ref = wav_input.squeeze().detach().cpu().numpy().astype(np.float32)
    
    def compute_zone_nSDP(mic_indices):
        nSDR_list = []
        for m in mic_indices:
            deg = p_C[m, :].detach().cpu().numpy().astype(np.float32)
            # Truncate/pad to match reference length
            min_len = min(len(ref), len(deg))
            s_target = ref[:min_len]
            s_est = deg[:min_len]

            denominator = np.sum(s_target**2)
            numerator = np.sum((s_target - s_est)**2)
            if denominator == 0:
                nSDR_val = 1e10
            else:
                nSDR_val = 10 * np.log10(numerator / denominator)
            nSDR_list.append(nSDR_val)
        nSDR_list = np.array(nSDR_list)
        return np.mean(nSDR_list)
    
    mean_B = compute_zone_nSDP(bright_zone_mics_index)
    mean_D = compute_zone_nSDP(dark_zone_mics_index)
    
    return mean_B,mean_D

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
        return np.mean(stoi_list)

    mean_B = compute_zone_stoi(bright_zone_mics_index)
    mean_D = compute_zone_stoi(dark_zone_mics_index)

    return mean_B,mean_D

def MSE(true_filter: torch.Tensor, predicted_filter: torch.Tensor):
    true_flat = true_filter.flatten()
    pred_flat = predicted_filter.flatten()
    if true_flat.dtype != pred_flat.dtype:
        pred_flat = pred_flat.to(true_flat.dtype)
    return torch.mean((true_flat - pred_flat) ** 2)

def MSEP(true_filter: torch.Tensor, predicted_filter: torch.Tensor,
                                  rir_test: torch.Tensor, 
                                  x_input: torch.Tensor, B_idx: list,D_idx: list) -> torch.Tensor:
    """
    Compute MSPE (Mean Squared Pressure Error) only in the Bright Zone (B_idx)
    between the desired pressure (from test filter/RIR) and the predicted pressure 
    (from candidate filter/train RIR).
    """
    
    # 1. Calculate Desired Pressure (Reference: Test Filter + Test RIR)
    p_des_full = compute_pressure_with_input(rir_test, true_filter, x_input) # [n_mics, n_samples]
    p_des_B = p_des_full[B_idx] # [M_B, n_samples]
    p_des_D = p_des_full[D_idx]

    # 2. Calculate Predicted Pressure (Candidate: Candidate Filter + Train RIR)
    p_pred_full = compute_pressure_with_input(rir_test, predicted_filter, x_input) # [n_mics, n_samples]
    p_pred_B = p_pred_full[B_idx] # [M_B, n_samples]
    p_pred_D = p_pred_full[D_idx]

    # 3. Compute MSE
    msep_loss_B = torch.mean((p_pred_B - p_des_B) ** 2)
    msep_loss_D = torch.mean((p_pred_D - p_des_D) ** 2)
    return msep_loss_B, msep_loss_D

def L_4_loss(q_true, q_pred, rir, x_input, H, bright_indices, dark_indices):
    M_B = len(bright_indices)
    M_D = len(dark_indices)
    fcentres = torch.tensor([1000, 2000])
    fd = torch.tensor(2**(1/6))
    delta_f = dgs.fs_target/dgs.J
    L_4 = 0
    for freq in fcentres:
        f_low = freq/fd
        f_high = freq*fd
        g = torch.fft.fft(q_pred, axis = 0)
        p_des_full = compute_pressure_with_input(rir, q_true, x_input)
        p_des_B = p_des_full[bright_indices]
        p_des_D = p_des_full[dark_indices]
        
        # AC_des is generally calculated in terms of pressure magnitude difference or ratio (in linear scale)
        E_des_B = torch.sum(p_des_B ** 2)
        E_des_D = torch.sum(p_des_D ** 2)
        AC_des = (M_D / M_B) * (E_des_B / E_des_D) if E_des_D.item() != 0 else torch.tensor(1e10)

        k_low = int(torch.ceil(f_low/delta_f))
        k_high = int(torch.ceil(f_high/delta_f))
        L_4_ = 0
        for k in range(k_low, k_high):
            AC_sim = AC_tilde(H[bright_indices][:,:,k], H[dark_indices][:,:,k], g[:,k], M_B, M_D)
            w_AC = w_ac(freq, ref_frequency=100, beta=1, min_weight=1)
            C = C_i(AC_des, w_AC, AC_sim)
            L_4_ += C**2
        L_4 += torch.sqrt(L_4_)
        del L_4_
    return L_4

def total_loss(true_filter,predicted_filter,rir_test,wav_input,B_idx,D_idx):
    mse_loss=MSE(true_filter,predicted_filter)
    cosine_loss=1-compute_cosine_similarity(true_filter,predicted_filter)
    msep_loss_B, msep_loss_D=MSEP(true_filter,predicted_filter,rir_test,wav_input,B_idx,D_idx)
    MSPE_loss=msep_loss_B
    H, freqs = compute_H_matrix(rir_test)
    AC_loss=L_4_loss(true_filter, predicted_filter, rir_test, x_input, H, B_idx, D_idx)
    return 1/4*(mse_loss+cosine_loss+MSPE_loss+AC_loss)

import matplotlib.pyplot as plt
import numpy as np

def plot_performance_metrics(AC, PESQ_B, PESQ_D, NSDR_B, NSDR_D, STOI_B, STOI_D,total_loss):
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
        "STOI": [STOI_B, STOI_D],
        "Total Loss": [total_loss]
    }
    
    zone_labels = {
        "AC (dB)": ["All zones"],
        "PESQ": ["Bright", "Dark"],
        "nSDR (dB)": ["Bright", "Dark"],
        "STOI": ["Bright", "Dark"],
        "Total Loss": ["Bright"]
    }
    
    plt.figure(figsize=(12, 10))
    
    for i, (metric_name, data) in enumerate(metrics.items(), 1):
        plt.subplot(3, 2, i)
        plt.boxplot(data, labels=zone_labels[metric_name])
        plt.title(metric_name)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def average_performance_metrics_with_filters(RIR_test, selected_filters, wav_input, bright_zone_mics_index_test, dark_zone_mics_index_test,true_filter):
    """
    Computes AC, PESQ, nSDR, and STOI for the entire test set using selected filters.
    
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
    NSDR_B_list, NSDR_D_list = [], []
    STOI_B_list, STOI_D_list = [], []
    tot_loss_list = []

    for i in range(RIR_test.shape[0]):
        print(f"\n--- Evaluating sample {i+1}/{RIR_test.shape[0]} ---")
        rirs = RIR_test[i]           # [n_mics, n_srcs, n_rir_samples]
        n_srcs = 3
        filter_len = 1024
        filters_flat = selected_filters[i]  # [3072]
        filters = filters_flat.reshape(n_srcs, filter_len)  # [3, 1024]
        true_filters_flat=true_filter[i]
        true_filters=true_filters_flat.reshape(n_srcs,filter_len)


        # Compute pressures with filters
        p_C = compute_pressure_with_input(rirs, filters, wav_input)

        # Compute metrics
        AC_i = float(acoustic_contrast(p_C, bright_zone_mics_index_test, dark_zone_mics_index_test))
        mean_pesq_B, mean_pesq_D = compute_pesq_unfiltered(p_C, wav_input, bright_zone_mics_index_test, dark_zone_mics_index_test)
        mean_NSDR_B, mean_NSDR_D = compute_nSDP(p_C, wav_input, bright_zone_mics_index_test, dark_zone_mics_index_test)
        mean_STOI_B, mean_STOI_D = compute_STOI(p_C, wav_input, bright_zone_mics_index_test, dark_zone_mics_index_test)
        total_loss_i=total_loss(true_filters,filters,rirs,wav_input,bright_zone_mics_index_test,dark_zone_mics_index_test)

        # Append results
        AC_list.append(AC_i)
        pesq_B_list.append(mean_pesq_B)
        pesq_D_list.append(mean_pesq_D)
        NSDR_B_list.append(mean_NSDR_B)
        NSDR_D_list.append(mean_NSDR_D)
        STOI_B_list.append(mean_STOI_B)
        STOI_D_list.append(mean_STOI_D)
        tot_loss_list.append(total_loss_i)

    # Convert to numpy arrays
    AC_list = np.array(AC_list)
    pesq_B_list = np.array(pesq_B_list)
    pesq_D_list = np.array(pesq_D_list)
    NSDR_B_list = np.array(NSDR_B_list)
    NSDR_D_list = np.array(NSDR_D_list)
    STOI_B_list = np.array(STOI_B_list)
    STOI_D_list = np.array(STOI_D_list)
    tot_loss_list = np.array(tot_loss_list)

    # Compute statistics
    results = {
        "AC": (np.mean(AC_list), np.min(AC_list), np.max(AC_list)),
        "PESQ_B": (np.mean(pesq_B_list), np.min(pesq_B_list), np.max(pesq_B_list)),
        "PESQ_D": (np.mean(pesq_D_list), np.min(pesq_D_list), np.max(pesq_D_list)),
        "NSDR_B": (np.mean(NSDR_B_list), np.min(NSDR_B_list), np.max(NSDR_B_list)),
        "NSDR_D": (np.mean(NSDR_D_list), np.min(NSDR_D_list), np.max(NSDR_D_list)),
        "STOI_B": (np.mean(STOI_B_list), np.min(STOI_B_list), np.max(STOI_B_list)),
        "STOI_D": (np.mean(STOI_D_list), np.min(STOI_D_list), np.max(STOI_D_list)),
        "Total Loss":(np.mean(tot_loss_list), np.min(tot_loss_list),   np.max(tot_loss_list))
    }

    # Optional: plot metrics
    plot_performance_metrics(AC_list, pesq_B_list, pesq_D_list, NSDR_B_list, NSDR_D_list, STOI_B_list, STOI_D_list,tot_loss_list)

    return results
# Assuming `selected_filters_test` comes from your random selection
results = average_performance_metrics_with_filters(RIRs_test, selected_filters, x_input, bright_zone_mics_index, dark_zone_mics_index,filters_test)

print(f"AC (mean, min, max): {results['AC']}")
print(f"PESQ Bright Zone (mean, min, max): {results['PESQ_B']}")
print(f"PESQ Dark Zone (mean, min, max): {results['PESQ_D']}")
print(f"NSDR Bright Zone (mean, min, max): {results['NSDR_B']}")
print(f"NSDR Dark Zone (mean, min, max): {results['NSDR_D']}")
print(f"STOI Bright Zone (mean, min, max): {results['STOI_B']}")
print(f"STOI Dark Zone (mean, min, max): {results['STOI_D']}")
print(f"Total loss Bright Zone (mean, min, max):{results['total_loss']}")
