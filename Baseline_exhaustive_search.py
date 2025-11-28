import torch, time, os
import torch.nn.functional as F
import numpy as np
import Loss_functions as LF
import Dataset_class as dc
from sklearn.neighbors import KDTree 
from scipy.io import wavfile
from torch.utils.data import DataLoader
from tqdm import tqdm
from Test_train_split import load_test_train_data





# ---- 2. Baseline: Extensive Brute-Force Search ----
def Extensive_search(
    filters_test: torch.Tensor,
    dictionary: torch.Tensor,
    IR_train: torch.Tensor,
    x_input: torch.Tensor,
    bright_zone_mics_index,
    dark_zone_mics_index,
    max_filters
):
    """
    Brute-force search over dictionary filters.
    - max_filters: hvor mange dictionary-filtre der skal testes mod (fra starten)
    """
    N_test = max_filters
    chosen_indices = []
    times_per_test = []
    per_filter_times = []

    # Loss weights
    lamda_mse, lambda_cosine, lambda_ac, lambda_msep = 0.25, 0.25, 0.25, 0.25
    print("\nStarting Extensive Brute-Force Search with FULL COMPOSITE LOSS...")
    for i in range(N_test):

        start_test = time.time()

        tf = filters_test[i].reshape(1, -1)
        tf2D = tf.reshape(L, J)

        min_loss = float("inf")
        best_idx = -1
        
        # --- Loop over dictionary filters ---
        for j in range(len(dictionary)):
            print(f"Test {i+1}/{N_test}, Dictionary Filter {j+1}/{len(dictionary)}", end="\r")
            

            df = dictionary[j].reshape(1, -1)
            df2D = df.reshape(L, J)
            rir_j = IR_train[j]

            # Compute losses
            mse_loss = LF.MSE(tf, df)
            cosine_loss = LF.Cosine_similarity(tf, df)
            H = LF.compute_H_matrix(rir_j)[0].to(device)
            ac_loss = LF.AC_loss(tf2D, df2D, H, bright_zone_mics_index, dark_zone_mics_index)
            msep_loss = LF.MSEP(tf2D, df2D, rir_j, x_input, bright_zone_mics_index, dark_zone_mics_index)[0]

            combined = (
                lamda_mse * mse_loss
                + lambda_cosine * cosine_loss
                + lambda_ac * ac_loss
                + lambda_msep * msep_loss
            )

            

            if combined.item() < min_loss:
                min_loss = combined.item()
                best_idx = j
        # Statistikker
        chosen_indices.append(best_idx)
        times_per_test.append(time.time() - start_test)


        print(
            f"Test {i+1}/{N_test}: "
            f"best filter = {best_idx}, "
            f"loss = {min_loss:.6f}, "
            f"time per test sample = {np.mean(times_per_test):.6f}s"
        )

    avg_time_per_test = np.mean(times_per_test)
    os.makedirs("Saved Filters", exist_ok=True)
    with open("Saved Filters/baseline_filters_time.txt", "a") as f:
        f.write(f"Chosen indices: {chosen_indices}\n")
        f.write(f"Average time per test sample: {avg_time_per_test:.6f}s\n")
    filter_q = filters_train[chosen_indices].to(device)
    torch.save(filter_q, "Saved Filters/baseline_filters.pt")
    return chosen_indices, avg_time_per_test

# ---- 3. K-20 ANN Search and Refinement (MODIFIED) ---
def ANN_Search_and_Refine(
    filters_test: torch.Tensor, dictionary: torch.Tensor, IR_train: torch.Tensor, IR_test: torch.Tensor, 
    x_input: torch.Tensor, k_neighbors: int = 20, max_filters: int = None
):  
    print("\nStarting K-20 ANN Search and Refinement with FULL COMPOSITE LOSS...")    
    # 1. Build the K-D Tree Index on the dictionary filters (y_train)
    dictionary_np = dictionary.cpu().numpy()
    filters_test_np = filters_test.cpu().numpy()
    tree = KDTree(dictionary_np, leaf_size=30)
    N_test = len(filters_test)
    
    # 2. Perform Approximate Search (Find indices of top K neighbors by Euclidean distance)
    _, indices = tree.query(filters_test_np, k=k_neighbors) 
    indices_tensor = torch.tensor(indices, device=dictionary.device)
    
    total_combined_loss = 0.0
    best_indices = []

    # Define the weights from the 'new' script (0.25 for each)
    lamda_mse, lambda_cosine, lambda_ac, lambda_msep = 0.25, 0.25, 0.25, 0.25

    # Timers for the first iteration (i=0)
    mse_times, cosine_times, ac_times, msep_times = [], [], [], []

    start_time = time.time()
    # 3. Refinement Loop: Calculate complex 4-component loss only on k_neighbors
    for i in range(N_test):
        test_filter_i_flat = filters_test[i].unsqueeze(0) # [1, L*J]
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
            mse_loss = LF.MSE(test_filter_i_flat, candidate_filter_j_flat)
            if i == 0: mse_times.append(time.time() - t_start)

            # 2. Cosine Similarity Loss (Filter Coefficients)
            t_start = time.time() if i == 0 else 0
            cosine_loss = LF.Cosine_similarity(test_filter_i_flat, candidate_filter_j_flat)
            if i == 0: cosine_times.append(time.time() - t_start)
            
            # 3. AC Loss (Acoustic Contrast - requires input and train/test RIR)
            t_start = time.time() if i == 0 else 0
            H = LF.compute_H_matrix(rir_train_j)[0].to(device)
            ac_loss = LF.AC_loss(test_filter_i_reshaped, candidate_filter_j_reshaped, H, bright_zone_mics_index, dark_zone_mics_index)
            if i == 0: ac_times.append(time.time() - t_start)
            
            # 4. MSEP Loss (Mean Squared Pressure Error - requires input and train/test RIR)
            t_start = time.time() if i == 0 else 0
            msep_loss = LF.MSEP(test_filter_i_reshaped, candidate_filter_j_reshaped, rir_train_j, x_input, bright_zone_mics_index, dark_zone_mics_index)[0]
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
    filter_q = filters_train[best_indices].to(device)
    end_time = time.time()
    #avg_baseline_loss = baseline_loss.item()
    #torch.save(filter_q, "Saved Filters/baseline_filters.pt")
    print(f"\n--- ANN Baseline Results (K=20, Composite Loss) ---")
    print(f"Total Test Samples Processed: {len(best_indices)}")
    #print(f"Average Combined Loss (across all samples): {avg_baseline_loss:.6f}")
    print(f"Average Search Time: {(end_time - start_time)/max_filters:.4f} seconds")
    print(f"\nIndices of All Chosen Filters (from y_train):")
    print(best_indices)
    return baseline_loss, best_indices

# ---- 4. Execution ----
if __name__ == "__main__":
    # Dummy check for fcentres
    #fcentres = torch.tensor([1000, 2000], device=device) # Example
    max_filters = 540

    # --- CONFIGURATION ---
    dark_zone_mics_index = [0,1,2,3,4,5,6,7,8,9,10,11]
    bright_zone_mics_index = [12]
    L = 3       # Loudspeaker (Sources)
    J = 1024    # Filter order
    # fs_target (used in loss functions) is assumed to be available in dgs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    data_dir = "Data Archive"

    data_test, data_train = load_test_train_data(test_size=0.25, random_seed=42)
    filters_test, filters_train = data_test[1], data_train[1]
    x_input = torch.tensor([1])
    IR_train = data_train[5]
    
    print(IR_train)
    Extensive_search(
        filters_test=filters_test, 
        dictionary=filters_train, 
        IR_train=IR_train, 
        x_input=x_input,
        bright_zone_mics_index=bright_zone_mics_index,
        dark_zone_mics_index=dark_zone_mics_index,
        max_filters=max_filters 
    )
    """
    ANN_Search_and_Refine(
        filters_test=y_test, 
        dictionary=y_train, 
        IR_train=IR_train, 
        IR_test=IR_test, 
        x_input=x_input,
        k_neighbors=20,
        max_filters=max_filters
    )
    """

