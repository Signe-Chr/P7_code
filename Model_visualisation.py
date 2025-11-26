import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, fftconvolve
# Assuming the file 'Dataset_generator_script.py' exists and is imported correctly
import Dataset_generator_script as dgs 
import os
from Dataset_generator_script import room_indices as ri
from Dataset_class import CustomDataset, L, J
from torch.utils.data import DataLoader

#load q
baseline_filters = np.array(torch.load(r"C:\Users\marst\OneDrive\Skrivebord\UNI\S. 7\PROJEKT\P7\Saved Filters\baseline_filters.pt"))
classification_filters = np.array(torch.load(r"C:\Users\marst\OneDrive\Skrivebord\UNI\S. 7\PROJEKT\P7\Saved Filters\classification_filters.pt"))
interpolation_filters = np.array(torch.load(r"C:\Users\marst\OneDrive\Skrivebord\UNI\S. 7\PROJEKT\P7\Saved Filters\interpolation_filters.pt"))
random_selection_filters = np.array(torch.load(r"C:\Users\marst\OneDrive\Skrivebord\UNI\S. 7\PROJEKT\P7\Saved Filters\random_selection_filters.pt"))
regression_filters = np.array(torch.load(r"C:\Users\marst\OneDrive\Skrivebord\UNI\S. 7\PROJEKT\P7\Saved Filters\regression_filters.pt"))


#---Load data and split into test and traning data---
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

srcs_pos_train = temp_var_train[7]
srcs_pos_test = temp_var_test[7]


dark_zone_mics_index=[0,1,2,3,4,5,6,7,8,9,10,11]
bright_zone_mics_index=[12]

# ================================================================
# 3. 2D Pressure Field Visualization with Bright/Dark Zones
# ================================================================
def pressure_field_2d(room_dim, sources, q_opt, center, fs=16000, grid_res=40, J=1024, r_zone=dgs.dark_mic_radius):
    """
    Compute and visualize a 2D SPL field given filter coefficients,
    and compute contrast between bright and dark zones.
    Bright zone: circle of radius r_zone around 'center'.
    Dark zone: everything outside that circle.
    """
    L = len(sources)
    q_opt = np.array(q_opt).reshape(L, J)

    # 2D grid (z = center[2])
    x = np.linspace(0, room_dim[0], grid_res)
    y = np.linspace(0, room_dim[1], grid_res)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.full_like(X, center[2])

    pressure_field = np.zeros_like(X)
    
    # Check if dgs.wav exists, otherwise use a default sine wave signal for simulation
    try:
        test_signal = dgs.wav
    except AttributeError:
        # Placeholder signal: 0.5 second, 1 kHz sine wave
        T = np.arange(0, 0.5, 1/fs)
        test_signal = np.sin(2 * np.pi * 1000 * T).astype(np.float32)
        print("Warning: Using placeholder sine wave as 'dgs.wav' was not found.")


    for i in range(grid_res):
        for j in range(grid_res):
            point = np.array([X[i, j], Y[i, j], Z[i, j]])
            p = 0
            for l, src in enumerate(sources):
                r = np.linalg.norm(point - np.array(src))
                
                # Simple free-field delay and attenuation (approximation)
                delay = int(r * fs / 343)
                h = np.zeros(J + 256)
                if delay < len(h):
                    h[delay] = 1.0 / (r + 1e-6)
                
                # Apply FIR filter (q_opt[l]) and then simulate propagation (h)
                filtered = lfilter(q_opt[l], 1, test_signal)
                out_l = fftconvolve(filtered, h) ##################################################################
                
                # Compute RMS pressure contribution
                p += np.sqrt(np.mean(out_l**2))
            
            pressure_field[i, j] = p

    # Normalize and convert to dB
    pressure_dB = 20 * np.log10( pressure_field / (np.max(pressure_field) + 1e-12) )


    # ------------------------------------------------------------
    # Bright and dark zones
    # ------------------------------------------------------------
    center_bright = np.array(center[:2])
    dist_bright = np.sqrt((X - center_bright[0])**2 + (Y - center_bright[1])**2)

    bright_mask = dist_bright <= r_zone
    dark_mask = dist_bright > r_zone

    # Avoid division by zero if masks are empty
    avg_bright = np.mean(pressure_field[bright_mask]) if np.any(bright_mask) else 1e-12
    avg_dark = np.mean(pressure_field[dark_mask]) if np.any(dark_mask) else 1e-12
    contrast_db = 20 * np.log10((avg_bright + 1e-12) / (avg_dark + 1e-12))

    print(f"Average Bright Zone Pressure: {avg_bright:.4f}")
    print(f"Average Dark Zone Pressure:   {avg_dark:.4f}")
    print(f"Bright/Dark Contrast:         {contrast_db:.2f} dB")

    # ------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 6))
    im = plt.imshow(pressure_dB.T, origin='lower',
                     extent=[0, room_dim[0], 0, room_dim[1]],
                     cmap='inferno', aspect='auto')
    plt.colorbar(im, label='SPL [dB]')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    #plt.title('Predicted Sound Pressure Field (SoftFilterNet)')

    # Plot speakers
    spk_x = [s[0] for s in sources]
    spk_y = [s[1] for s in sources]
    plt.scatter(spk_x, spk_y, c='cyan', s=80, edgecolors='black', label='Speakers', zorder=2)

    # Plot bright zone circle
    theta = np.linspace(0, 2*np.pi, 200)
    plt.plot(center_bright[0] + r_zone*np.cos(theta),
             center_bright[1] + r_zone*np.sin(theta),
             'w--', label='Bright Zone', zorder=1)

    plt.legend()
    plt.tight_layout()
    plt.show()



test_index = 0


pressure_field_2d(list(X_test[test_index][6:]), srcs_pos_test[0], filters_test[0], list(X_test[test_index][3:6]), fs=16000, grid_res=40, J=1024, r_zone=dgs.dark_mic_radius)

