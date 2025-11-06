import torch
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree 
import time
import Dataset_generator_script as dgs
from scipy.io import wavfile
import Loss_functions as LF
import torch.nn as nn

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


# ---- 4. K-20 ANN Search and Refinement (MODIFIED) ----

def ANN_Search_and_Refine(
    test_filters: torch.Tensor, dictionary: torch.Tensor, IR_train: torch.Tensor, IR_test: torch.Tensor, 
    fcentres: torch.Tensor, M_B: int, M_D: int, x_input: torch.Tensor, k_neighbors: int = 20
):
    N_test = len(test_filters)
    L_1_loss = nn.MSELoss()
    
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
            mse_loss = L_1_loss(test_filter_i_flat, candidate_filter_j_flat)
            if i == 0: mse_times.append(time.time() - t_start)

            # 2. Cosine Similarity Loss (Filter Coefficients)
            t_start = time.time() if i == 0 else 0
            cosine_loss = LF.L_2_loss(test_filter_i_flat, candidate_filter_j_flat)
            if i == 0: cosine_times.append(time.time() - t_start)
            
            # 3. AC Loss (Acoustic Contrast - requires input and train/test RIR)
            t_start = time.time() if i == 0 else 0
            print(test_filter_i_flat.shape)
            ac_loss = LF.L_3_loss(test_filter_i_flat, candidate_filter_j_flat, rir_test_i, rir_train_j, x_input, [12])
            if i == 0: ac_times.append(time.time() - t_start)
            
            # 4. MSEP Loss (Mean Squared Pressure Error - requires input and train/test RIR)
            t_start = time.time() if i == 0 else 0
            H = LF.compute_H_matrix(rir_train_j)
            msep_loss = LF.L_4_loss(
                test_filter_i_reshaped, rir_train_j,
                x_input, H, [0,1,2,3,4,5,6,7,8,9,10,11], [12]) #L_4_loss(q_opt, rir, x_input, H, bright_indices, dark_indices)
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