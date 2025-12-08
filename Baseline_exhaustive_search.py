import torch, time, os
import torch.nn.functional as F
import numpy as np
import Loss_functions as LF
import Dataset_class as dc
from sklearn.neighbors import KDTree 
from scipy.io import wavfile
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


    # Loss weights
    lamda_mse, lambda_cosine, lambda_ac, lambda_msep = 0.002, 0.005, 0.009, 2.859 # 1, 1, 1, 1 #1/3.729, 1/1.000, 1/18.390, 1/17.181
    print("\nStarting Extensive Brute-Force Search with FULL COMPOSITE LOSS...")
    for i in tqdm(range(N_test)):

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
            rir_j = IR_train[j].to(torch.double)

            # Compute losses
            mse_loss = LF.MSE(df, tf)
            cosine_loss = LF.Cosine_similarity(df, tf)
            H = LF.compute_H_matrix(rir_j)[0].to(device)
            ac_loss = LF.AC_loss(df2D, tf2D, H, bright_zone_mics_index, dark_zone_mics_index)
            msep_loss = LF.MSEP(df2D, tf2D, rir_j, x_input, bright_zone_mics_index, dark_zone_mics_index)[0]
            
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
    torch.save(filter_q, "Saved Filters Speech/baseline_filters_speech.pt")
    return chosen_indices, avg_time_per_test



# ---- 4. Execution ----
if __name__ == "__main__":
    # --- CONFIGURATION ---
    dark_zone_mics_index = [0,1,2,3,4,5,6,7,8,9,10,11]
    bright_zone_mics_index = [12]
    L = 3       # Loudspeaker (Sources)
    J = 1024    # Filter order
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    

    data_test, data_train, data_val = load_test_train_data(val_size=0.10, random_seed=42)
    filters_test, filters_train = data_val[1], data_train[1]
                
    max_filters = len(filters_test)
    x_input = torch.tensor([1])
    IR_train = data_train[5]
    
    Extensive_search(
        filters_test=filters_test, 
        dictionary=filters_train, 
        IR_train=IR_train, 
        x_input=x_input,
        bright_zone_mics_index=bright_zone_mics_index,
        dark_zone_mics_index=dark_zone_mics_index,
        max_filters=max_filters 
    )

