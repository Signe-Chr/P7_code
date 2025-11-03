import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Loss_functions_baseline import compute_pressure_with_input,L_2_loss_with_input,AC_loss_with_input
from scipy.io import wavfile
np.random.seed(69420)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
wav_path = "relaxing-guitar-loop-v5-245859.wav"
fs_wav, wav = wavfile.read(wav_path)
if wav.ndim > 1:
    wav = np.mean(wav, axis=1)
wav = wav[5*fs_wav : 7*fs_wav]
wav = wav / np.max(np.abs(wav))  # scale to [-1,1]
x_input = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
x_input = x_input.to(device)

# ---- 1. Load data from VAST archive
data = np.load("VAST_filter_archive.npy", allow_pickle=True).item()
X_list, filters_list,rt_60_list,rir_list,B_idx,D_idx = [], [],[], [], [], []

for key, inner in data.items():
    # Robust handling of features
    rt60 = inner.get('RT60', 0.0)
    phone_tilt = inner.get('Phone_tilt', 0.0)
    user_orient = inner.get('User_orientation', 0.0)
    spatial = np.array(inner.get('Spatial_position', [0, 0, 0]), dtype=np.float32).ravel()
    rir_array = inner.get('IR')

    # Input feature vector
    X = np.concatenate([[rt60], [phone_tilt], [user_orient], spatial])
    X_list.append(X)

    # q_matrix = target filter coefficients
    q = inner.get('q_matrix', np.zeros(3072, dtype=np.float32))
    filters_list.append(q.flatten())
    
    rt_60_list.append(rt60)
    
    rir_list.append(rir_array)
    
    bright_idx = np.array(inner.get('bright_zone_mics_index'), dtype=np.int64)
    dark_idx = np.array(inner.get('dark_zone_mics_index'), dtype=np.int64)
    B_idx.append(bright_idx)
    D_idx.append(dark_idx)
# Convert lists of numpy arrays to GPU tensors
B_idx = [torch.tensor(b, device=device) for b in B_idx]
D_idx = [torch.tensor(d, device=device) for d in D_idx]

# ---- 2. Prepare arrays and tensors
X = np.stack(X_list).astype(np.float32)        # [N_total, num_features]
filters = np.stack(filters_list).astype(np.float32)  # [N_total, filter_length]
rir=np.stack(rir_list).astype(np.float32)
rt60_array=np.array(rt_60_list)
num_total, input_size = X.shape
filter_length = filters.shape[1]


configs_tensor = torch.from_numpy(X)        # [N_total, num_features]
filters_tensor = torch.from_numpy(filters)  # [N_total, filter_length]
rir_tensor = torch.from_numpy(rir)


configs_tensor = configs_tensor.to(device)
filters_tensor = filters_tensor.to(device)
rir_tensor = rir_tensor.to(device)
fcentre = torch.tensor([1000, 2000], device=device)

print(f"Configs shape: {configs_tensor.shape}")
print(f"Filters shape: {filters_tensor.shape}")
print(f"RIRs shape: {rir_tensor.shape}")
# ---- 3. Split into train/test sets
unique_rooms = np.unique(rt60_array)
print("All rooms (RT60 values):", unique_rooms)


def Exhaustive_MSE(test, dictionary):
    diffs=test.unsqueeze(1) - dictionary.unsqueeze(0)
    mse_matrix = torch.mean(diffs ** 2, dim=2)  # [N_test, N_train]
    return mse_matrix

def Exhaustive_cosine_similarity(test, dictionary):
    y_train_norm = F.normalize(dictionary, p=2, dim=1)
    y_test_norm = F.normalize(test, p=2, dim=1)
    similarity_matrix = torch.mm(y_test_norm, y_train_norm.T)
    cosine_distances_matrix = 1 - similarity_matrix
    return cosine_distances_matrix


def Exhaustive_MSEP_with_input(test_filters, dictionary, RIR_train, RIR_test, input_signal):
    """
    Compute MSPE including the input signal x.

    Parameters:
        test_filters: [N_test, n_srcs, filter_len]
        dictionary: [N_dict, n_srcs, filter_len]
        RIR_train: [N_dict, n_mics, n_srcs, n_rir]
        RIR_test: [N_test, n_mics, n_srcs, n_rir]
        input_signal: [n_srcs, n_input_samples] (torch.Tensor)

    Returns:
        msep_matrix: [N_test, N_dict] torch.Tensor
    """
    N_test = len(test_filters)
    N_dict = len(dictionary)
    msep_matrix = torch.zeros((N_test, N_dict), dtype=torch.float32)

    for i, test_filter in enumerate(test_filters):
        rir_test_i = RIR_test[i]  # [n_mics, n_srcs, n_samples]
        g_ref = test_filter.unsqueeze(0) if test_filter.ndim == 1 else test_filter
        # Desired pressure
        p_des = compute_pressure_with_input(rir_test_i, g_ref, input_signal)

        for j, candidate in enumerate(dictionary):
            rir_train_j = RIR_train[j]
            g_pred = candidate.unsqueeze(0) if candidate.ndim == 1 else candidate
            p_pred = compute_pressure_with_input(rir_train_j, g_pred, input_signal)
            mse = torch.mean((p_pred - p_des) ** 2)
            msep_matrix[i, j] = mse

    return msep_matrix

def Exhaustive_AC_with_input(test_filters, dictionary, RIR_train, RIR_test, B_idx, D_idx, fcentres, x_input):
    """
    Compute AC loss for all test filters vs dictionary, using real input.
    """
    N_test = len(test_filters)
    N_dict = len(dictionary)
    ac_loss_matrix = torch.zeros((N_test, N_dict), dtype=torch.float32)

    for i, test_filter in enumerate(test_filters):
        rir_test_i = RIR_test[i]
        bright_idx = B_idx[i]
        dark_idx = D_idx[i]

        # Compute reference AC from test filter
        AC_des = AC_loss_with_input(rir_test_i, bright_idx, dark_idx, test_filter.unsqueeze(0), x_input)

        # Compare test RIR to all dictionary candidates
        for j, candidate in enumerate(dictionary):
            rir_train_j = RIR_train[j]
            ac_loss = L_2_loss_with_input(candidate, fcentres, rir_train_j, bright_idx, dark_idx, x_input, AC_des=AC_des)
            ac_loss_matrix[i, j] = ac_loss

    return ac_loss_matrix


def loss_function_with_input(test_filters, dictionary, lamda_mse, lambda_cosine,
                             lambda_ac, lambda_msep, RIR_train, RIR_test,
                             B_idx, D_idx, fcentre, input_signal):
    loss_matrix = (
        lamda_mse * Exhaustive_MSE(test_filters, dictionary)
        + lambda_cosine * Exhaustive_cosine_similarity(test_filters, dictionary)
        + lambda_ac * Exhaustive_AC_with_input(test_filters, dictionary, RIR_train, RIR_test, B_idx, D_idx, fcentre,input_signal)
        + lambda_msep * Exhaustive_MSEP_with_input(test_filters, dictionary, RIR_train, RIR_test, input_signal)
    )

    best_loss_per_test = torch.min(loss_matrix, dim=1).values
    baseline_loss = best_loss_per_test.mean().item()
    return baseline_loss


baseline_losses = []

for test_room in unique_rooms:  # <-- loop over unique rooms
    train_mask = rt60_array != test_room
    test_mask = rt60_array == test_room

    # Apply masks
    X_train = configs_tensor[train_mask]
    X_test = configs_tensor[test_mask]
    y_train = filters_tensor[train_mask]
    y_test = filters_tensor[test_mask]
    RIR_train = rir_tensor[train_mask]
    RIR_test = rir_tensor[test_mask]
    print("Training samples:", X_train.shape[0])
    print("Test samples (unseen room):", X_test.shape[0])

    baseline_loss = loss_function_with_input(
    test_filters=y_test, 
    dictionary=y_train, 
    lamda_mse=0.25, 
    lambda_cosine=0.25, 
    lambda_ac=0.25, 
    lambda_msep=0.25,
    RIR_train=RIR_train, 
    RIR_test=RIR_test, 
    B_idx=B_idx, 
    D_idx=D_idx, 
    fcentre=fcentre,
    input_signal=x_input    
    )
    baseline_losses.append(baseline_loss)
    print(baseline_loss)

avg_baseline_loss = sum(baseline_losses) / len(baseline_losses)
print(f'Baseline loss averaged over {len(baseline_losses)} trials with wavfile input: {avg_baseline_loss}')

