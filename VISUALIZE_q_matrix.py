import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.io import wavfile
import time
import pyroomacoustics as pra
import os

example_key = "VAST_example_1"
room_dim=[8.12, 7.35, 3.00]

def load_q_matrix(archive_path, key_name):
    """
    Loads a specific q_matrix from the persistent archive file using its key name.

    Args:
        archive_path (str): Path to the .npy archive file.
        key_name (str): The unique key used to identify the matrix when archiving.

    Returns:
        np.ndarray: The requested q_matrix, or None if the file or key is not found.
    """
    if not os.path.exists(archive_path):
        print(f"Error: Archive file not found at {archive_path}.")
        return None

    try:
        # 1. Load the entire dictionary from the file
        loaded_data = np.load(archive_path, allow_pickle=True)
        
        # 2. Check structure (ensure it's a 0-D array containing the dict)
        if loaded_data.ndim == 0:
            archive_dict = loaded_data.item()
        else:
            print(f"Error: File at {archive_path} is not a valid dictionary archive.")
            return None
        
        # 3. Retrieve the specific matrix by key
        if key_name not in archive_dict:
            print(f"Error: Key '{key_name}' not found in archive.")
            print(f"Available keys: {list(archive_dict.keys())}")
            return None
            
        print(f"Successfully loaded q_matrix for key: '{key_name}' with shape {archive_dict[key_name].shape}")
        return archive_dict[key_name]

    except Exception as e:
        print(f"An error occurred while loading or unpacking the archive: {e}")
        return None

q_matrix = load_q_matrix("VAST_filter_archive.npy", example_key)


def visualize_pressure_field(q_matrix, wav_path, fs_target, sources, room_dim, 
                             grid_res=50, z_plane=1.5):
    """
    Computes and plots the RMS pressure field using a direct-path approximation 
    for visualization purposes.
    """
    L = q_matrix.shape[0]

    # Load test signal (use a short segment for visualization speed)
    fs_wav, wav = wavfile.read(wav_path)
    if wav.ndim > 1: wav = wav[:, 0]
    wav = np.array(wav, dtype=float)
    wav = wav / (np.max(np.abs(wav)) + 1e-12)
    test_signal = wav[:fs_target // 4] if len(wav) >= fs_target // 4 else wav
    
    x_grid = np.linspace(0, room_dim[0], grid_res)
    y_grid = np.linspace(0, room_dim[1], grid_res)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    Gx, Gy = X.shape
    pressure_field = np.zeros_like(X, dtype=float)
    
    speed_of_sound = 343.0

    print("\nComputing pressure field (coarse grid for speed)...")
    tstart = time.perf_counter()

    for ix in range(Gx):
        for iy in range(Gy):
            point = np.array([X[ix,iy], Y[ix,iy], z_plane])
            p_sum_sq = 0.0 # Using squared sum for coherent approximation (simpler here)
            
            # Sum the pressure contribution from each loudspeaker
            
            # --- CONVOLVE DRIVE SIGNALS (d_l) ---
            # d_l = x * q_l
            drives = [fftconvolve(test_signal, q_matrix[l])[:len(test_signal)] for l in range(L)]
            
            # --- CONVOLVE WITH PATH (h_l) ---
            total_pressure = np.zeros_like(drives[0])
            for l in range(L):
                src_pos = np.array(sources[l])
                r = np.linalg.norm(point - src_pos)
                
                # Approximate acoustic path (direct path only 1/r)
                if r > 1e-6:
                    h_val = 1.0 / r 
                else:
                    h_val = 1.0 
                    
                # Time delay
                delay = int(round(r * fs_target / speed_of_sound))
                
                # Apply delay and gain
                out_l = np.roll(drives[l], delay) * h_val
                out_l[:delay] = 0.0 # Zero out wrapped part

                total_pressure += out_l
            
            # RMS value of the total pressure waveform
            pressure_field[ix,iy] = np.sqrt(np.mean(total_pressure**2))
    
    print("Pressure computed in {:.2f}s".format(time.perf_counter() - tstart))

    # Compute averages for bright/dark masks
    room_center_x = room_dim[0] / 2
    bright_mask = X < room_center_x
    dark_mask = ~bright_mask
    
    avg_bright = np.mean(pressure_field[bright_mask])
    avg_dark = np.mean(pressure_field[dark_mask])
    
    print(f"Average pressure (bright) = {avg_bright:.6f} ; (dark) = {avg_dark:.6f}")
    contrast_db = 20.0 * np.log10((avg_bright + 1e-12) / (avg_dark + 1e-12))
    print(f"Contrast (bright/dark) [dB] = {contrast_db:.2f}")

    # Plot relative SPL
    P_db = 20.0 * np.log10(pressure_field / (np.max(pressure_field) + 1e-12) + 1e-12)
    
    plt.figure(figsize=(8,6))
    plt.imshow(P_db.T, origin='lower', extent=[0, room_dim[0], 0, room_dim[1]], cmap='inferno', aspect='auto')
    plt.colorbar(label='Relative dB')
    plt.scatter([s[0] for s in sources], [s[1] for s in sources], c='cyan', marker='*', s=100, edgecolors='k', label='Loudspeakers')
    
    # Add Bright/Dark zone markers
    plt.plot([room_center_x, room_center_x], [0, room_dim[1]], 'w--', linewidth=1, label='Zone Boundary')
    plt.text(room_center_x / 2, room_dim[1] * 0.9, 'Bright Zone', color='white', ha='center', fontsize=10, weight='bold')
    plt.text(room_center_x + (room_dim[0] - room_center_x) / 2, room_dim[1] * 0.9, 'Dark Zone', color='white', ha='center', fontsize=10, weight='bold')

    plt.title('Relative SPL (dB) at z={:.2f} m (VAST Filter Result)'.format(z_plane))
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()
    plt.tight_layout()
    plt.show()

visualize_pressure_field(q_matrix, "Signe_sang.wav", 16000, )