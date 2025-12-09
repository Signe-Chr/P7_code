
import torch, time, os
import numpy as np
import Loss_functions as LF
#from sklearn.neighbors import KDTree 
from tqdm import tqdm
from Test_train_split import load_test_train_data, x_input, L, J



# --- CONFIGURATION ---
dark_zone_mics_index = [0,1,2,3,4,5,6,7,8,9,10,11]
bright_zone_mics_index = [12]
L = 3       # Loudspeaker (Sources)
J = 1024    # Filter order
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

save_dir = "Saved Filters Speech"
os.makedirs(save_dir, exist_ok=True)

data_test, data_train, data_val = load_test_train_data(val_size=0.10, random_seed=42)
filters_test, filters_train = data_test[1], data_train[1]
        
max_filters = len(filters_test)
IR_train = data_train[5]
indeces = [14, 99, 124, 124, 124, 49, 124, 77, 10, 14, 40, 14, 124, 22, 40, 154, 70, 237, 217, 9, 14, 237, 14, 217, 82, 77, 22, 14, 2, 14, 86, 14, 22, 14, 14, 40, 40, 40, 174, 88, 14, 14, 154, 70, 99, 205, 14, 237, 184, 14, 124, 40, 40, 9, 124, 124, 249, 237, 40, 244, 14, 40, 2, 124, 77, 14, 14, 29, 14, 77, 14, 2, 29, 174, 124, 174, 14, 154, 14, 174, 99, 90, 40, 29, 14, 237, 237, 14, 237, 14, 14, 154, 9, 125, 14, 90, 184, 124, 86, 249, 244, 14, 237, 126, 14, 124, 124, 124, 40, 14, 237, 14, 57, 77, 77, 184, 88, 99, 14, 22, 22, 124, 5, 10, 14, 9, 184, 14, 14, 237, 184, 124, 10, 40, 90, 217, 29, 14, 2, 77, 125, 174, 174, 99]
filter_q = filters_train[indeces].to(device)
torch.save(filter_q, save_dir + "/baseline_filters_speech.pt")