import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.io import wavfile
import time
import pyroomacoustics as pra
import os

example_key = "ACC_key_(4, 0, 0, 0)"


# --- Corrected Load Function ---
def load_q_matrix(archive_path, key_name):
    """
    Loads a specific q_matrix and the full item dictionary from the persistent archive file.

    Args:
        archive_path (str): Path to the .npy archive file.
        key_name (str): The unique key used to identify the matrix when archiving.

    Returns:
        tuple: (q_matrix (np.ndarray), item_dict (dict)) 
               The requested q_matrix and the full dictionary item 
               ({'q_matrix':..., 'parameters':...}), or (None, None) if not found.
    """
    if not os.path.exists(archive_path):
        print(f"Error: Archive file not found at {archive_path}.")
        return None, None

    try:
        loaded_data = np.load(archive_path, allow_pickle=True)
        
        if loaded_data.ndim == 0:
            archive_dict = loaded_data.item()
        else:
            print(f"Error: File at {archive_path} is not a valid dictionary archive.")
            return None, None
        
        if key_name not in archive_dict:
            print(f"Error: Key '{key_name}' not found in archive.")
            print(f"Available keys: {list(archive_dict.keys())}")
            return None, None
            
        # --- FIX IS HERE: Access the entire item dict and the q_matrix ---
        loaded_item_dict = archive_dict[key_name]
        q_matrix = loaded_item_dict.get('q_matrix')

        if q_matrix is None:
             print(f"Error: Archive item for key '{key_name}' is corrupt or missing 'q_matrix'.")
             return None, None

        print(f"Successfully loaded q_matrix for key: '{key_name}' with shape {q_matrix.shape}")
        
        # Return the matrix AND the full item dictionary, as requested
        return q_matrix, loaded_item_dict

    except Exception as e:
        print(f"An error occurred while loading or unpacking the archive: {e}")
        return None, None


q = load_q_matrix("ACC_filter_archive.npy", example_key)

arpath = "ACC_filter_archive.npy"


def visualize_pressure_field(q_matrix, wav_path, fs_target, sources, room_dim, center_pos, R_mic, grid_res=300, z_plane=1.5):
    """
    Computes and plots the RMS pressure field using a direct-path approximation
    for visualization purposes, with zones correctly defined by the circular array.
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

    # --- CORRECT ZONE MASKING (CIRCULAR) ---
    center_x, center_y = center_pos[0], center_pos[1]
    dist_sq = (X - center_x)**2 + (Y - center_y)**2
    
    # Bright Zone: Inside or on the boundary of the microphone circle
    bright_mask = dist_sq <= (R_mic + 0.1)**2 # Added 0.1 buffer for visualization contrast
    dark_mask = dist_sq > (R_mic + 0.1)**2 
    
    avg_bright = np.mean(pressure_field[bright_mask])
    avg_dark = np.mean(pressure_field[dark_mask])
    
    print(f"Average pressure (bright) = {avg_bright:.6f} ; (dark) = {avg_dark:.6f}")
    contrast_db = 10.0 * np.log10((avg_bright + 1e-12) / (avg_dark + 1e-12))
    print(f"Contrast (bright/dark) [dB] = {contrast_db:.2f}")

    # Plot relative SPL
    P_db = 20.0 * np.log10(pressure_field / (np.max(pressure_field) + 1e-12) + 1e-12)
    
    plt.figure(figsize=(8,6))
    plt.imshow(P_db.T, origin='lower', extent=[0, room_dim[0], 0, room_dim[1]], cmap='inferno', aspect='auto')
    plt.colorbar(label='Relative dB')
    plt.scatter([s[0] for s in sources], [s[1] for s in sources], c='cyan', marker='*', s=100, edgecolors='k', label='Loudspeakers')
    
    # Add Circular Bright/Dark zone markers
    theta = np.linspace(0, 2 * np.pi, 100)
    boundary_x = center_x + R_mic * np.cos(theta)
    boundary_y = center_y + R_mic * np.sin(theta)
    plt.plot(boundary_x, boundary_y, 'w--', linewidth=1, label='Zone Boundary')
    
    # Place text labels
    plt.text(center_x, center_y, 'Bright Zone', color='white', ha='center', fontsize=10, weight='bold')
    plt.text(center_x + R_mic + 0.2, center_y + R_mic + 0.2, 'Dark Zone', color='white', ha='left', fontsize=10, weight='bold')


    plt.title('Relative SPL (dB) at z={:.2f} m (VAST Filter Result)'.format(z_plane))
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()
    plt.tight_layout()
    plt.show()




wav_path = "Signe_sang.wav"
out_q_path = "Signe_sang_pos.wav"


def sound_output(pos, length_sec, archive_path = arpath, key_name=example_key, wav_path=wav_path, output_wav_name="output_sound.wav"):
    """
    Generates and returns the audio waveform at a specific position (pos) 
    in the room, given a VAST filter and source audio.
    
    Args:
        pos (list or np.array): [x, y, z] coordinates of the listening point.
        length_sec (float): Duration of the output sound to generate (in seconds).
        archive_path (str): Path to the .npy archive file.
        key_name (str): Key of the filter to load.
        wav_path (str): Path to the input source audio file.
        output_wav_name (str): Filename to save the resulting audio.
        
    Returns:
        np.ndarray: The resulting audio waveform (normalized).
    """
    print(f"\n--- Generating sound output at position {pos} for {length_sec} seconds ---")
    
    # 1. Load VAST filter and parameters
    q_matrix, params = load_q_matrix(archive_path, key_name)
    if q_matrix is None:
        return np.array([])
    
    # Extract necessary parameters
    sources = params.get('sources_position', [])
    fs_target = params.get('fs_target', 16000)
    L = q_matrix.shape[0]

    if not sources or L != len(sources):
        print("Error: Source positions not found or mismatch filter size.")
        return np.array([])
    
    speed_of_sound = 343.0
    
    # 2. Load and trim source audio
    try:
        fs_wav, wav = wavfile.read(wav_path)
    except FileNotFoundError:
        print(f"Error: Source audio file not found at {wav_path}.")
        return np.array([])
        
    # Resample or ensure correct rate if necessary (for now, just use fs_target)
    if fs_wav != fs_target:
        print(f"Warning: Source audio fs ({fs_wav}) differs from target fs ({fs_target}). Using source fs for simplicity.")
        fs_target = fs_wav
        
    if wav.ndim > 1: wav = wav[:, 0]
    wav = np.array(wav, dtype=float)
    wav = wav / (np.max(np.abs(wav)) + 1e-12) # Normalize source signal

    n_samples = int(length_sec * fs_target)
    input_signal = wav[:n_samples]

    if len(input_signal) == 0:
        print("Error: Input signal length is zero.")
        return np.array([])

    # 3. Apply VAST filters (drives) and calculate acoustic path
    
    # Calculate filtered drives for each speaker: d_l = x * q_l
    print("Applying VAST filters (Convolution)...")
    drives = [fftconvolve(input_signal, q_matrix[l]) for l in range(L)]
    
    # Ensure all drives are the same length for summation (trim to shortest)
    max_drive_len = min(len(d) for d in drives)
    total_pressure = np.zeros(max_drive_len)
    
    point = np.array(pos)
    
    print("Simulating direct path propagation...")
    
    for l in range(L):
        src_pos = np.array(sources[l])
        r = np.linalg.norm(point - src_pos)
        
        # Approximate acoustic path (direct path only 1/r gain)
        h_val = 1.0 / (r + 1e-6) # Add small epsilon to prevent division by zero
        
        # Time delay (in samples)
        delay = int(round(r * fs_target / speed_of_sound))
        
        # Apply filter and trim to correct length
        drive_l = drives[l][:max_drive_len]

        # Apply delay and gain
        out_l = np.roll(drive_l, delay) * h_val
        out_l[:delay] = 0.0 # Zero out wrapped part

        total_pressure += out_l
        
    # 4. Normalize and Save Output
    if np.max(np.abs(total_pressure)) > 1e-12:
        output_waveform = total_pressure / np.max(np.abs(total_pressure))
    else:
        output_waveform = total_pressure # Silence

    # Convert to 16-bit integer for saving as WAV
    output_int16 = (output_waveform * 32767).astype(np.int16)

    try:
        wavfile.write("output_" + output_wav_name, fs_target, output_int16)
        print(f"Successfully saved output audio to 'output_{output_wav_name}' at {fs_target} Hz.")
    except Exception as e:
        print(f"Warning: Could not save output WAV file. {e}")


    return output_waveform

if __name__ == "__main__":

    visualize_pressure_field(q[1]["q_matrix"], "Signe_sang.wav", 16000, q[1]['sources_position'], q[1]['room_dim'], q[1]['Center'], q[1]['R'])


    sound_output([5, 5, 5], 2)