import os, sys, torch
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from Loss_functions import MSE, Cosine_similarity, MSEP, AC_loss, compute_H_matrix
from performance_evaluation_unfiltered import compute_pressure_with_input as cpwi
from performance_evaluation_unfiltered import load_wav_file
from Test_train_split import load_test_train_data, load_wav_file, L, J, x_input, indeces_bright, indeces_dark
from pesq import pesq
from pystoi import stoi



def load_data_and_model(chosen_model, filters_dir = "Saved Filters Speech/"):
    if chosen_model == "random":
        data_random_selection = torch.load("Saved Filters/random_filters.pt")
        model = data_random_selection['selected_filters']
    else:
        model = torch.load(f"{filters_dir + chosen_model}_filters.pt")


    #---Load data and split into test and traning data---
    data_test, data_train, data_val = load_test_train_data(data_dir="Data Archive Speech")

    filters_test=data_test[1]
    filters_train=data_train[1]

    if chosen_model == "acc":
        model = filters_test
    
    n_srcs_test=data_test[4]
    n_srcs_train=data_train[4]
    
    RIRs_test=data_test[5]
    RIRs_train=data_train[5]
    
    return n_srcs_test, n_srcs_train, filters_test, filters_train, RIRs_test, RIRs_train, model

def compute_pressure_with_input(rir: torch.Tensor, filter_q: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    n_mics, n_srcs, n_rir_samples = rir.shape
    filter_len = filter_q.shape[1]
    n_input_samples = reference.shape[-1]
    # The total combined impulse response length (h_combined) is n_rir_samples + filter_len - 1
    # The final pressure length (p) is h_combined_len + n_input_samples - 1
    output_len = n_rir_samples + filter_len + n_input_samples - 2
    
    p = torch.zeros((n_mics, output_len), device=rir.device)

    for m in range(n_mics):
        p_m = torch.zeros(output_len, device=rir.device)
        for s in range(n_srcs):
            # Combined filter impulse response: h_combined = RIR * filter_q (via standard convolution)
            rir_m_s = rir[m, s, :].unsqueeze(0).unsqueeze(0).float()  # cast to float32
            q_s = filter_q[s, :].unsqueeze(0).unsqueeze(0).float()    # cast to float32

            h_combined = F.conv1d(rir_m_s, q_s, padding=q_s.shape[-1]-1)
            
            p_m_s = F.conv1d(reference.reshape((1, 1, n_input_samples)).float(), h_combined, padding=h_combined.shape[-1]-1).squeeze()
            
            p_m += p_m_s
        p[m, :] = p_m
    
    return p

#--------------------------------------------------------------
# Performance metric functions:
# Acoustic Contrast, PESQ, NSDP, STOI, total loss, attenuation
# -------------------------------------------------------------
def acoustic_contrast(p_C, indeces_bright, indeces_dark):
    p_B=p_C[indeces_bright]
    p_D=p_C[indeces_dark]
    e_B=torch.sum(p_B**2)
    e_D=torch.sum(p_D**2)
    M_B=len(indeces_bright)
    M_D=len(indeces_dark)
    AC=(M_D / M_B) * (e_B / e_D) if e_D.item() != 0 else torch.tensor(1e10)
    return AC

def loss_functions(true_filter, predicted_filter, rir_test, wav_input, B_idx, D_idx):
    L = 3
    J = 1024
    mse_loss = MSE(predicted_filter, true_filter)
    cosine_loss = Cosine_similarity(predicted_filter.reshape(1, L*J), true_filter.reshape(1, L*J))
    msep_loss_B, _ = MSEP(predicted_filter, true_filter, rir_test, wav_input, B_idx, D_idx)
    MSPE_loss = msep_loss_B
    H, _ = compute_H_matrix(rir_test)
    AC_los = AC_loss(predicted_filter, true_filter, H, B_idx, D_idx)
    return (mse_loss+cosine_loss+MSPE_loss+AC_los), [mse_loss, cosine_loss, MSPE_loss, AC_los]

def attenuation(rir, raw_wav, filtered, zone):
    raw_signal = cpwi(rir, raw_wav)[zone]
    e_raw = torch.sum(raw_signal**2)
    e_filt = torch.sum(filtered**2)
    return e_raw/e_filt

def compute_nSDP(p_C: torch.Tensor, wav_input: torch.Tensor, indeces_bright, rir: torch.Tensor):
    p_B = p_C[indeces_bright]
    #ref = wav_input.float()
    #ref = F.pad(wav_input.float(), (3, 0))
    
    #d_B_list = []
    #rir_m = rir[indeces_bright] 
    #max_len = ref.shape[-1] + rir_m.shape[-1] - 1
    #mic_pressure = torch.zeros((1, max_len), device=rir.device)
    #pad = len(rir_m[:,0].T)-1
    #for s in range(rir_m.shape[1]):
     #   conv_result = F.conv1d(ref.unsqueeze(0).unsqueeze(0), rir_m[:,s].unsqueeze(0), padding=pad)
      #  conv_result = conv_result.squeeze(0)
       # conv_result = F.pad(conv_result, (0, max_len - conv_result.shape[-1]))
       # mic_pressure += conv_result
    #mic_pressure[mic_pressure==0] += 1e-12
    #d_B_list.append(mic_pressure)
    d_B=cpwi(rir,wav_input)[indeces_bright]

    #d_B_tensor = torch.stack(d_B_list).squeeze(0)
    min_len = min(d_B.shape[-1], p_B.shape[-1])
    d_B = d_B[:, :min_len]
    p_B = p_B[:, :min_len]
    
    #rms_d_B_tensor = torch.sqrt(torch.mean(d_B_tensor ** 2))
    #rms_pB = torch.sqrt(torch.mean(p_B ** 2))
    #d_B_tensor = d_B_tensor * (rms_pB / rms_d_B_tensor)

    numerator = torch.sum((d_B - p_B) ** 2)
    denominator = torch.sum(d_B ** 2)

    return numerator / denominator

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
#--------------------------------------------------------------
# Main evaluation functions
# -------------------------------------------------------------
def average_performance_metrics_with_filters(RIR_test, selected_filters, wav_input, indeces_bright_test, indeces_dark_test, true_filter, chosen_model):
    """
    Computes AC, PESQ, NSDP, and STOI for the entire test set using selected filters.
    
    Parameters:
        RIR_test: [n_samples, n_mics, n_srcs, n_rir_samples] torch.Tensor
        selected_filters: [n_samples, n_srcs, filter_len] torch.Tensor
        wav_input: [1, n_input_samples] torch.Tensor
        indeces_bright_test: list of bright-zone mic indices
        indeces_dark_test: list of dark-zone mic indices
    
    Returns:
        Average, min, max of each metric for bright and dark zones
    """
    AC_list =[]
    NSDP_B_list= []
    attenuation_list = []
    attenuation_list_bz=[]
    STOI_B_list, STOI_D_list = [], []
    pesq_B_list, pesq_D_list = [], []

    for i in tqdm(range(RIR_test.shape[0]), disable=not sys.stdout.isatty()):
        print(i)
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
        AC_i = float(acoustic_contrast(p_C, indeces_bright_test, indeces_dark_test))
        mean_NSDP_B = compute_nSDP(p_C, wav_input, indeces_bright_test, rirs)
        atten = attenuation(rirs, wav_input, p_C[indeces_dark_test], indeces_dark_test)
        atten_bz=attenuation(rirs, wav_input, p_C[indeces_bright_test], indeces_bright_test)
        mean_pesq_B, mean_pesq_D = compute_pesq_unfiltered(p_C, wav_input, indeces_bright_test, indeces_dark_test)
        mean_STOI_B, mean_STOI_D = compute_STOI(p_C, wav_input, indeces_bright_test, indeces_dark_test)
    
        # Append results
        AC_list.append(AC_i)
        NSDP_B_list.append(mean_NSDP_B)
        attenuation_list.append(atten)
        attenuation_list_bz.append(atten_bz)
        pesq_B_list.append(mean_pesq_B)
        pesq_D_list.append(mean_pesq_D)
        STOI_B_list.append(mean_STOI_B)
        STOI_D_list.append(mean_STOI_D)

    # Convert to numpy arrays
    AC_list = np.array(AC_list)
    NSDP_B_list = np.array(NSDP_B_list)
    attenuation_arr = np.array(attenuation_list)
    attenuation_arr_bz = np.array(attenuation_list_bz)
    pesq_B_list = np.array(pesq_B_list)
    pesq_D_list = np.array(pesq_D_list)
    STOI_B_list = np.array(STOI_B_list)
    STOI_D_list = np.array(STOI_D_list)
    
    # Compute statistics
    results = {
        "AC": (np.sqrt(np.var(10*np.log10(AC_list))), 10*np.log10(np.mean(AC_list)),  np.min(10*np.log10(AC_list)), np.max(10*np.log10(AC_list))),
        "NSDP_B": (np.sqrt(np.var(10*np.log10(NSDP_B_list))).item(), 10*np.log10(np.mean(NSDP_B_list)).item(), 10*np.log10(np.min(NSDP_B_list)).item(), 10*np.log10(np.max(NSDP_B_list)).item()),
        "Attenuation_DZ": (np.sqrt(np.var(10*np.log10(attenuation_arr))).item(),10*np.log10( np.mean(attenuation_arr).item()), np.min(10*np.log10(attenuation_arr)).item(), np.max(10*np.log10(attenuation_arr)).item()),
        "Attenuation_BZ": (np.sqrt(np.var(10*np.log10(attenuation_arr_bz))).item(), 10*np.log10(np.mean(attenuation_arr_bz)).item(), np.min(10*np.log10(attenuation_arr_bz)).item(), np.max(10*np.log10(attenuation_arr_bz)).item()),
        "PESQ_BZ": (np.sqrt(np.var(pesq_B_list)).item(),np.mean(pesq_B_list).item(), np.min(pesq_B_list),np.max(pesq_B_list)),
        "PESQ_DZ": (np.sqrt(np.var(pesq_D_list)).item(),np.mean(pesq_D_list).item(), np.min(pesq_D_list),np.max(pesq_D_list)),
        "STOI_BZ": (np.sqrt(np.var(STOI_B_list)).item(),np.mean(STOI_B_list).item(), np.min(STOI_B_list),np.max(STOI_B_list)),
        "STOI_DZ": (np.sqrt(np.var(STOI_D_list)).item(),np.mean(STOI_D_list).item(), np.min(STOI_D_list),np.max(STOI_D_list)),
    }
    print(f"AC (std, mean, min, max): {results['AC']}")
    print(f"NSDP Bright Zone (std, mean, min, max): {results['NSDP_B']}")
    print(f"Attenuation Dark Zone (std, mean, min, max):{results['Attenuation_DZ']}")
    print(f"Attenuation Bright Zone (std, mean, min, max):{results['Attenuation_BZ']}")
    print(f"PESQ Bright Zone (std, mean, min, max):{results['PESQ_BZ']}")
    print(f"PESQ Dark Zone (std, mean, min, max):{results['PESQ_DZ']}")
    print(f"STOI Bright Zone (std, mean, min, max):{results['STOI_BZ']}")
    print(f"STOI Dark Zone (std, mean, min, max):{results['STOI_DZ']}")
    
    return results

def loss_function_evaluation(RIR_test, selected_filters, wav_input, indeces_bright_test, indeces_dark_test, true_filter):
    """
    Computes loss functions and attenuation for the entire test set using selected filters.
    
    Parameters:
        RIR_test: [n_samples, n_mics, n_srcs, n_rir_samples] torch.Tensor
        selected_filters: [n_samples, n_srcs, filter_len] torch.Tensor
        wav_input: [1, n_input_samples] torch.Tensor
        indeces_bright_test: list of bright-zone mic indices
        indeces_dark_test: list of dark-zone mic indices
    
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
        total_loss_i, indiv_losses = loss_functions(true_filters, filters, rirs, wav_input, indeces_bright_test, indeces_dark_test)
        
        # Append results
        tot_loss_list.append(total_loss_i)
        individual_losses.append(indiv_losses)

    # Convert to numpy arrays
    tot_loss_list = np.array(tot_loss_list)
    individual_losses_arr = np.array(individual_losses)
    
    # Compute statistics
    results = {
        "Total Loss":(np.sqrt(np.var(tot_loss_list)).item(),np.mean(tot_loss_list).item(), np.min(tot_loss_list).item(),   np.max(tot_loss_list).item()),
        "Individual Losses" : (np.sqrt(individual_losses_arr.var(axis=0)),individual_losses_arr.mean(axis=0), individual_losses_arr.min(axis=0), individual_losses_arr.max(axis=0)),
    }

    print(f"Total loss Bright Zone (std, mean, min, max):{results['Total Loss']}")
    print(f"Individual Losses (MSE, Cosine, MSEP, AC) (std, mean, min, max):{results['Individual Losses']}")

    return results

'Choose between:'
"random"
"baseline"
"interpolation"
"regression"
"classification"
"acc"

chosen_model = "baseline"
print(f"Du har valgt {chosen_model} til evaluering.")

x_input = x_input
n_srcs_test, n_srcs_train, filters_test, filters_train, RIRs_test, RIRs_train, model = load_data_and_model(chosen_model)

results = average_performance_metrics_with_filters(RIRs_test, model, x_input, indeces_bright, indeces_dark, filters_test, chosen_model)
loss = loss_function_evaluation(RIRs_test, model, x_input, indeces_bright, indeces_dark, filters_test)
#results.append(loss)

gemt = [f"AC (std, mean, min, max): {results['AC']}\n",
            f"NSDP Bright Zone (std, mean, min, max): {results['NSDP_B']}\n", 
            f"Attenuation Dark Zone (std, mean, min, max):{results['Attenuation_DZ']}\n",
            f"Attenuation Bright Zone (std, mean, min, max):{results['Attenuation_BZ']}\n",
            f"Total loss Bright Zone (std, mean, min, max):{loss['Total Loss']}\n",
            f"PESQ Bright Zone (std, mean, min, max):{results['PESQ_BZ']}\n",
            f"PESQ Dark Zone (std, mean, min, max):{results['PESQ_DZ']}\n",
            f"STOI Bright Zone (std, mean, min, max):{results['STOI_BZ']}\n",
            f"STOI Dark Zone (std, mean, min, max):{results['STOI_DZ']}\n",
            f"Individual Losses (MSE, Cosine, MSEP, AC) (std, mean, min, max):{loss['Individual Losses']}"]
            

with open(f"Performance Evaluation/Results/{chosen_model}.txt", "w") as file:
    for string in gemt:
        file.writelines(string)
