import numpy as np
import random
import sys
import os
import glob

#---Load data and split into test and traning data---
#data_dir="Signes_data"
#full_data = os.listdir(data_dir)
#data_points = []
#train_points = []
#test_points = []
#for data in full_data:
#    i = int(data.split("_")[1])
#    if (i in ri) and (i not in ri[::4]):
#        train_points.append(data)
#        data_points.append(data)
#    else:
#        test_points.append(data)
#        data_points.append(data)
#        
#data_train=CustomDataset(data_dir,train_points)
#data_train_loader=DataLoader(data_train,batch_size=len(data_train), shuffle=False)
#data_test=CustomDataset(data_dir,test_points)
#data_test_loader=DataLoader(data_test,batch_size=len(data_test), shuffle=False)
#
#temp_var_train=[batch for batch in data_train_loader][0]
#temp_var_test=[batch for batch in data_test_loader][0]
#
#X_train=temp_var_train[0]
#X_test=temp_var_test[0]
#
#filters_train=temp_var_train[1]
#filters_test=temp_var_test[1]
#
#bright_zone_mics_index_train=temp_var_train[2]
#bright_zone_mics_index_test=temp_var_test[2]
#
#dark_zone_mics_index_train=temp_var_train[3]
#dark_zone_mics_index_test=temp_var_test[3]
#
#n_srcs_train=temp_var_train[4]
#n_srcs_test=temp_var_test[4]
#
#RIRs_train=temp_var_train[5]
#RIRs_test=temp_var_test[5]





# --- CONFIGURATION ---
# Assumed dimensions based on your original script:
L = 3     # Loudspeaker (Sources)
J = 1024  # Filter order

def select_random_filter(folder_path="VAST_filter_archive_730"):
    """
    Searches the specified folder for NPY filter files, selects one at random, 
    loads the filter's dictionary content, and extracts the data.
    """
    print(f"--- 1. Searching Filter Archive Folder: {folder_path} ---")
    
    # Use glob to find all .npy files in the specified folder
    file_paths = glob.glob(os.path.join(folder_path, "*.npy"))
    
    if not file_paths:
        print(f"WARNING: No NPY files found in folder '{folder_path}'.")
        # Create a dummy data dictionary if the file structure doesn't exist
        L_dummy = 3; J_dummy = 1024
        dummy_q = np.random.rand(L_dummy, J_dummy).astype(np.float32)
        dummy_ir = np.random.rand(10, L_dummy, 512).astype(np.float32)
        
        chosen_key = "DUMMY_FILTER_FALLBACK"
        chosen_data = {
            'q_matrix': dummy_q, 
            'IR': dummy_ir, 
            'RT60': 0.5, 
            'room_dim': [5, 4, 3], 
            'Phone_tilt': 0.0,
            'User_orientation': 0.0
        }
        print(f"Using dummy filter data: {chosen_key}")
    else:
        # --- 2. Random Selection and Loading ---
        chosen_file_path = random.choice(file_paths)
        # The filter key is derived from the filename
        chosen_key = os.path.basename(chosen_file_path).replace(".npy", "")
        
        print(f"Successfully found {len(file_paths)} filter files.")
        print(f"Randomly selected file path: {chosen_file_path}")
        
        try:
            # Load the individual filter dictionary from the selected file
            # Assuming each NPY file contains a single dictionary entry.
            chosen_data = np.load(chosen_file_path, allow_pickle=True).item()
        except Exception as e:
            print(f"Error loading data from {chosen_file_path}. Is the file corrupted? Error: {e}")
            return chosen_key, 'Error Loading', 'Error Loading'

    # --- 3. Extraction and Reporting ---
    chosen_q_matrix = chosen_data.get('q_matrix', 'N/A')
    chosen_IR = chosen_data.get('IR', 'N/A')
    
    print("\n--- 2. Chosen Filter Configuration ---")
    print(f"**Randomly Selected Key:** {chosen_key}")
    
    # Report associated metadata
    print("\n--- 3. Extracted Metadata ---")
    print(f"RT60: {chosen_data.get('RT60', 'N/A')}")
    print(f"Room Dimensions: {chosen_data.get('room_dim', 'N/A')}")
    print(f"Phone Tilt: {chosen_data.get('Phone_tilt', 'N/A')}")
    
    # Report filter and IR shapes
    print("\n--- 4. Filter Data Shape ---")
    if isinstance(chosen_q_matrix, np.ndarray):
        print(f"Filter (q_matrix) shape: {chosen_q_matrix.shape}")
        print(f"Total Filter Coefficients (L*J): {chosen_q_matrix.size}")
    else:
        print(f"Filter (q_matrix) data: {chosen_q_matrix}")
        
    if isinstance(chosen_IR, np.ndarray):
        print(f"Impulse Response (IR) shape: {chosen_IR.shape}")
    else:
        print(f"Impulse Response (IR) data: {chosen_IR}")
        
    return chosen_key, chosen_q_matrix, chosen_IR

if __name__ == "__main__":
    # The default path is now "VAST_filter_archive_730"
    select_random_filter()