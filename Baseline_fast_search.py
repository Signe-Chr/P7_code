import torch
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree 
import time
import Dataset_generator_script as dgs
from scipy.io import wavfile

# --- CONFIGURATION ---
L = 3       # Loudspeaker (Sources)
J = 1024    # Filter order
# fs_target (used in loss functions) is assumed to be available in dgs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- 1. Load and Prepare Data ----

data = np.load("VAST_filter_archive.npy", allow_pickle=True).item()

# Get indices (M_B and M_D are needed for loss function)
bright_zone_mics_index = data["VAST_0_0_0_0"].get('bright_zone_mics_index', [])
dark_zone_mics_index = data["VAST_0_0_0_0"].get('dark_zone_mics_index', [])

M_B = len(bright_zone_mics_index)
M_D = len(dark_zone_mics_index)
n_srcs = L # Based on the L=3 setting

X_list, y_list, IR_list = [], [], []

for key, inner in data.items():
    rt60 = inner.get('RT60', 0.0)
    phone_tilt = inner.get('Phone_tilt', 0.0)
    user_orient = inner.get('User_orientation', 0.0)
    spatial = np.array(inner.get('Spatial_position', [0,0,0]), dtype=np.float32).ravel()
    room_dim = np.array(inner.get('room_dim', [0,0,0]), dtype=np.float32).ravel()
    IR = inner.get('IR', np.zeros((M_B + M_D, L, 1), dtype=np.float32)) 
    
    X = np.concatenate([
        [rt60], [phone_tilt], [user_orient], spatial, room_dim
    ])
    
    q_matrix = inner.get('q_matrix', np.zeros((L, J), dtype=np.float32)) 
    y = np.ravel(q_matrix)
    
    IR_list.append(IR)
    X_list.append(X)
    y_list.append(y)

X = np.stack(X_list).astype(np.float32)
y = np.stack(y_list).astype(np.float32)
IR_array = np.stack(IR_list).astype(np.float32) # [N_total, n_mics, n_srcs, n_samples]

# ---- 1b. Load Dummy Input Signal (REPLACE WITH REAL WAV LOADING) ----
# Since the WAV file is not available, we create a dummy input signal.
# Assumes x_input is [1, n_samples] and the max sample length (N) is IR_array.shape[-1]
if not hasattr(dgs, 'fs_target'): dgs.fs_target = 16000
N_samples = IR_array.shape[-1]
np.random.seed(42)
#dummy_input = np.random.randn(dgs.fs_target * 2) # 2 seconds of noise at 16kHz
#dummy_input = dummy_input[:N_samples] # Truncate to match RIR length
#dummy_input = dummy_input / np.max(np.abs(dummy_input)) # Normalize
wav_path = "relaxing-guitar-loop-v5-245859.wav"
fs_wav, wav = wavfile.read(wav_path)
if wav.ndim > 1:
    wav = np.mean(wav, axis=1)
wav = wav[5*fs_wav : 7*fs_wav]
wav = wav / np.max(np.abs(wav))  # scale to [-1,1]
x_input = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
x_input = x_input.to(device)
#x_input = torch.from_numpy(input_.astype(np.float32)).unsqueeze(0).to(device) # [1, n_input_samples]
#print(f"Using dummy input signal of length {x_input.shape[1]}")


# ---- 2. Train/Test Split and Scaling ----

scaler_X = StandardScaler()
X_train_indices, X_test_indices, _, _ = train_test_split(
    np.arange(X.shape[0]), np.arange(X.shape[0]), test_size=0.2, random_state=42
)

scaler_X.fit(X[X_train_indices]) 

y_train = torch.tensor(y[X_train_indices], dtype=torch.float32).to(device)
y_test = torch.tensor(y[X_test_indices], dtype=torch.float32).to(device)
IR_train = torch.tensor(IR_array[X_train_indices], dtype=torch.float32).to(device)
IR_test = torch.tensor(IR_array[X_test_indices], dtype=torch.float32).to(device)

print(f"Dictionary size (y_train): {y_train.shape}")
print(f"Query size (y_test): {y_test.shape}")

# ---- 3. Loss Functions (New Composite Metrics) ----

def compute_pressure_with_input(rir: torch.Tensor, filter_q: torch.Tensor, x_input: torch.Tensor) -> torch.Tensor:
    """
    Simulates the acoustic pressure at all mics by convolving RIRs and filters with the input signal.

    Parameters:
        rir: [n_mics, n_srcs, n_rir_samples]
        filter_q: [n_srcs, filter_len]
        x_input: [1, n_input_samples] (The source signal)
    
    Returns:
        p: [n_mics, n_output_samples] (Acoustic pressure)
    """
    n_mics, n_srcs, n_rir_samples = rir.shape
    filter_len = filter_q.shape[1]
    n_input_samples = x_input.shape[-1]
    # The total combined impulse response length (h_combined) is n_rir_samples + filter_len - 1
    # The final pressure length (p) is h_combined_len + n_input_samples - 1
    output_len = n_rir_samples + filter_len + n_input_samples - 2
    
    # Zero pad x_input for convolution
    x_input_padded = F.pad(x_input, (0, output_len - n_input_samples), 'constant', 0)
    p = torch.zeros((n_mics, output_len), device=rir.device)

    for m in range(n_mics):
        p_m = torch.zeros(output_len, device=rir.device)
        for s in range(n_srcs):
            # Combined filter impulse response: h_combined = RIR * filter_q (via standard convolution)
            rir_m_s = rir[m, s, :].unsqueeze(0).unsqueeze(0) # [1, 1, n_rir_samples]
            q_s = filter_q[s, :].unsqueeze(0).unsqueeze(0) # [1, 1, filter_len]
            
            # --- CRITICAL FIX: SWAP INPUT/KERNEL FOR CONV1D ---
            # Since n_rir_samples (512) < filter_len (1024), we must swap them for F.conv1d.
            # Convolution is commutative: rir * q = q * rir
            h_combined = F.conv1d(q_s, rir_m_s, padding=0).squeeze()
            
            # Convolve h_combined with input signal x (x_input) using FFT
            
            # Pad h_combined to ensure final output length matches 'output_len'
            h_combined_padded = F.pad(h_combined, (0, output_len - h_combined.shape[0]), 'constant', 0)
            
            n_fft = 2**int(np.ceil(np.log2(output_len)))
            
            H = torch.fft.rfft(h_combined_padded, n=n_fft)
            X_fft = torch.fft.rfft(x_input_padded, n=n_fft).squeeze(0)
            
            P_fft = H * X_fft
            p_m_s = torch.fft.irfft(P_fft, n=n_fft)[:output_len] # Back to time domain
            
            p_m += p_m_s
        p[m, :] = p_m
    
    return p

def Exhaustive_MSEP_with_input_single(test_filter_reshaped: torch.Tensor, candidate_filter_reshaped: torch.Tensor,
                                  rir_test: torch.Tensor, rir_train: torch.Tensor, 
                                  x_input: torch.Tensor, B_idx: list) -> torch.Tensor:
    """
    Compute MSPE (Mean Squared Pressure Error) only in the Bright Zone (B_idx)
    between the desired pressure (from test filter/RIR) and the predicted pressure 
    (from candidate filter/train RIR).
    """
    
    # 1. Calculate Desired Pressure (Reference: Test Filter + Test RIR)
    p_des_full = compute_pressure_with_input(rir_test, test_filter_reshaped, x_input) # [n_mics, n_samples]
    p_des_B = p_des_full[B_idx] # [M_B, n_samples]

    # 2. Calculate Predicted Pressure (Candidate: Candidate Filter + Train RIR)
    p_pred_full = compute_pressure_with_input(rir_train, candidate_filter_reshaped, x_input) # [n_mics, n_samples]
    p_pred_B = p_pred_full[B_idx] # [M_B, n_samples]

    # 3. Compute MSE
    msep_loss = torch.mean((p_pred_B - p_des_B) ** 2)
    return msep_loss

def Exhaustive_AC_with_input_single(candidate_filter_reshaped: torch.Tensor, rir_train: torch.Tensor, 
                                  rir_test: torch.Tensor, test_filter_reshaped: torch.Tensor,
                                  fcentres: torch.Tensor, x_input: torch.Tensor,
                                  M_B: int, M_D: int) -> torch.Tensor:
    """
    Compute AC loss using the train RIR and the candidate filter,
    comparing against the reference AC from the test RIR and test filter.
    This is the L_2_loss_with_input logic from the 'new' script.
    NOTE: This is a significantly simplified (and less robust) implementation
    of the original AC loss, as full frequency band logic is complex. 
    We calculate the overall broadband AC.
    """
    
    # 1. Calculate Reference AC (AC_des) from Test setup (using the energy of the input-driven pressure)
    p_des_full = compute_pressure_with_input(rir_test, test_filter_reshaped, x_input)
    p_des_B = p_des_full[bright_zone_mics_index]
    p_des_D = p_des_full[dark_zone_mics_index]
    
    # AC_des is generally calculated in terms of pressure magnitude difference or ratio (in linear scale)
    E_des_B = torch.sum(p_des_B ** 2)
    E_des_D = torch.sum(p_des_D ** 2)
    AC_des = (M_D / M_B) * (E_des_B / E_des_D) if E_des_D.item() != 0 else torch.tensor(1e10)
    
    # 2. Calculate Simulated AC (AC_sim) from Candidate setup
    p_pred_full = compute_pressure_with_input(rir_train, candidate_filter_reshaped, x_input)
    # --- FIX: Extract Bright Zone pressure before calculation
    p_pred_B = p_pred_full[bright_zone_mics_index] 
    p_pred_D = p_pred_full[dark_zone_mics_index]

    E_sim_B = torch.sum(p_pred_B ** 2)
    E_sim_D = torch.sum(p_pred_D ** 2)
    AC_sim = (M_D / M_B) * (E_sim_B / E_sim_D) if E_sim_D.item() != 0 else torch.tensor(1e10)
    
    # 3. Compute L_2 loss (error between desired and simulated AC)
    # Error is the violation of the target AC (AC_des)
    AC_loss = torch.max(torch.tensor(0.0, device=AC_sim.device), AC_des - AC_sim) ** 2
    
    return torch.sqrt(AC_loss)

def Exhaustive_Cosine_similarity_single(test_filter_flat: torch.Tensor, candidate_filter_flat: torch.Tensor):
    """Cosine distance between two flattened filters."""
    y_test_norm = F.normalize(test_filter_flat, p=2, dim=1)
    y_cand_norm = F.normalize(candidate_filter_flat, p=2, dim=1)
    similarity = torch.mm(y_test_norm, y_cand_norm.T)
    cosine_distance = 1 - similarity.squeeze()
    return cosine_distance

def MSE_distance_single(test_filter: torch.Tensor, candidate_filter: torch.Tensor):
    """Calculates MSE between two flattened filters: [1, L*J] vs [1, L*J]"""
    diff = test_filter - candidate_filter
    mse = torch.mean(diff ** 2)
    return mse

# ---- 4. K-20 ANN Search and Refinement (MODIFIED) ----

def ANN_Search_and_Refine(
    test_filters: torch.Tensor, dictionary: torch.Tensor, IR_train: torch.Tensor, IR_test: torch.Tensor, 
    fcentres: torch.Tensor, M_B: int, M_D: int, x_input: torch.Tensor, k_neighbors: int = 20
):
    N_test = len(test_filters)
    
    # 1. Build the K-D Tree Index on the dictionary filters (y_train)
    dictionary_np = dictionary.cpu().numpy()
    test_filters_np = test_filters.cpu().numpy()
    tree = KDTree(dictionary_np, leaf_size=30)
    
    # 2. Perform Approximate Search (Find indices of top K neighbors by Euclidean distance)
    _, indices = tree.query(test_filters_np, k=k_neighbors) 
    indices_tensor = torch.tensor(indices, device=dictionary.device)
    
    total_combined_loss = 0.0
    best_indices = []

    # Define the weights from the 'new' script (0.25 for each)
    lamda_mse, lambda_cosine, lambda_ac, lambda_msep = 0.25, 0.25, 0.25, 0.25

    # Timers for the first iteration (i=0)
    mse_times, cosine_times, ac_times, msep_times = [], [], [], []

    # 3. Refinement Loop: Calculate complex 4-component loss only on k_neighbors
    for i in range(N_test):
        test_filter_i_flat = test_filters[i].unsqueeze(0) # [1, L*J]
        test_filter_i_reshaped = test_filter_i_flat.reshape(L, J) # [L, J]
        rir_test_i = IR_test[i] # [n_mics, L, n_samples]
        
        k_indices = indices_tensor[i] # Indices of top K candidates in the *dictionary*
        k_dictionary = dictionary[k_indices] # [k_neighbors, L*J]
        k_IR_train = IR_train[k_indices] # [k_neighbors, n_mics, L, n_samples]
        
        min_combined_loss = float('inf')
        best_candidate_absolute_index = -1 

        # Loop over top K candidates for composite loss calculation
        for j in range(k_neighbors):
            candidate_filter_j_flat = k_dictionary[j].unsqueeze(0) # [1, L*J]
            candidate_filter_j_reshaped = candidate_filter_j_flat.reshape(L, J) # [L, J]
            rir_train_j = k_IR_train[j] # [n_mics, L, n_samples]
            
            # --- 4-COMPONENT COMPOSITE LOSS CALCULATION ---
            
            # 1. MSE Loss (Filter Coefficients)
            t_start = time.time() if i == 0 else 0
            mse_loss = MSE_distance_single(test_filter_i_flat, candidate_filter_j_flat)
            if i == 0: mse_times.append(time.time() - t_start)

            # 2. Cosine Similarity Loss (Filter Coefficients)
            t_start = time.time() if i == 0 else 0
            cosine_loss = Exhaustive_Cosine_similarity_single(test_filter_i_flat, candidate_filter_j_flat)
            if i == 0: cosine_times.append(time.time() - t_start)
            
            # 3. AC Loss (Acoustic Contrast - requires input and train/test RIR)
            t_start = time.time() if i == 0 else 0
            ac_loss = Exhaustive_AC_with_input_single(
                candidate_filter_j_reshaped, rir_train_j, rir_test_i, test_filter_i_reshaped,
                fcentres, x_input, M_B, M_D
            )
            if i == 0: ac_times.append(time.time() - t_start)
            
            # 4. MSEP Loss (Mean Squared Pressure Error - requires input and train/test RIR)
            t_start = time.time() if i == 0 else 0
            msep_loss = Exhaustive_MSEP_with_input_single(
                test_filter_i_reshaped, candidate_filter_j_reshaped,
                rir_test_i, rir_train_j, x_input, bright_zone_mics_index
            )
            if i == 0: msep_times.append(time.time() - t_start)
            
            combined_loss = (lamda_mse * mse_loss) + (lambda_cosine * cosine_loss) + \
                            (lambda_ac * ac_loss) + (lambda_msep * msep_loss)
            
            if combined_loss.item() < min_combined_loss:
                min_combined_loss = combined_loss.item()
                best_candidate_absolute_index = k_indices[j].item()
        
        total_combined_loss += min_combined_loss
        best_indices.append(best_candidate_absolute_index)
        print(f"Test Sample {i+1}/{N_test}: Chosen Dictionary Index: {best_candidate_absolute_index} (Min Loss: {min_combined_loss:.6f})")

        # Print times and break after the first iteration
        if i == 0:
            print("\n--- TIMING RESULTS (First Test Sample, Averaged over K=20 Candidates) ---")
            print(f"Average MSE Loss Time:   {np.mean(mse_times):.6f} s")
            print(f"Average Cosine Loss Time: {np.mean(cosine_times):.6f} s")
            print(f"Average AC Loss Time:     {np.mean(ac_times):.6f} s")
            print(f"Average MSEP Loss Time:  {np.mean(msep_times):.6f} s")
            print("---------------------------------------------------------------------------")
            #break # Stop the outer loop after the first test sample
            
    # Since we break after the first iteration, N_test is effectively 1 for the results below.
    baseline_loss = total_combined_loss / (i + 1)
    return baseline_loss, best_indices

# ---- 5. Execution ----

if __name__ == "__main__":
    # Dummy check for fcentres
    fcentres = torch.tensor([1000, 2000], device=device) # Example

    print("\nStarting K-20 ANN Search and Refinement with FULL COMPOSITE LOSS...")
    start_time = time.time()
    
    avg_baseline_loss, chosen_indices = ANN_Search_and_Refine(
        test_filters=y_test, 
        dictionary=y_train, 
        IR_train=IR_train, 
        IR_test=IR_test, 
        fcentres=fcentres, 
        M_B=M_B, 
        M_D=M_D,
        x_input=x_input, # Pass the input signal
        k_neighbors=20
    )
    
    end_time = time.time()
    
print(f"\n--- ANN Baseline Results (K=20, Composite Loss) ---")
print(f"Total Test Samples Processed: {len(chosen_indices)}")
print(f"Average Combined Loss (across all samples): {avg_baseline_loss:.6f}")
print(f"Total Search Time: {end_time - start_time:.4f} seconds")
print(f"\nIndices of All Chosen Filters (from y_train):")
print(chosen_indices)