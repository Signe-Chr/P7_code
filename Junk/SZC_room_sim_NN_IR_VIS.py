import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.signal import lfilter, fftconvolve
from scipy.io import wavfile
import os
import torch
import torch.nn as nn

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Signe_sang.wav")
fs, lyd_data = wavfile.read(file_path)

lyd_data = list(np.array(lyd_data[0:fs*1])/max(lyd_data))
print(f"Sample rate: {fs} Hz, length: {len(lyd_data):.2f}")

# ----------------------
# Room & array settings
# ----------------------
fs = 16000
room_dim = [6.0, 3.0, 3.5]
absorption = 0.2
max_order = 10

# Create a shoebox room
room = pra.ShoeBox(
    room_dim,
    fs=fs,
    materials=pra.Material(absorption),
    max_order=max_order,
)

# ----------------------
# Speakers (sources)
# ----------------------
sources = [
    [0.5, 0.0, 1.5], [1.0, 0.0, 1.5], [1.5, 0.0, 1.5],
    [2.0, 0.0, 1.5], [2.5, 0.0, 1.5], [3.0, 0.0, 1.5],
    [3.5, 0.0, 1.5], [4.0, 0.0, 1.5], [4.5, 0.0, 1.5],
    [5.0, 0.0, 1.5], [5.5, 0.0, 1.5],
]

for s in sources:
    room.add_source(s)

# ----------------------
# Microphones (not used for visualization, but needed for RIR computation if needed)
# ----------------------
mic_positions = []
for x in np.linspace(0.5, 2.5, 6):
    for y in np.linspace(0.5, 2.5, 4):
        mic_positions.append([x, y, 1.5])
for x in np.linspace(3.5, 5.5, 6):
    for y in np.linspace(0.5, 2.5, 4):
        mic_positions.append([x, y, 1.5])

mic_positions = np.array(mic_positions).T
room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))

# ----------------------
# Compute RIRs (optional - only if needed for your model)
# ----------------------
room.compute_rir()
IR = room.rir
n_srcs = len(sources)
n_mics = mic_positions.shape[1]
J = 512  # filter length

print(f"Number of sources: {n_srcs}")

def prepare_rir_input(IR, n_mics, n_srcs, max_length=512):
    """
    Prepare RIR data as CNN input tensor
    Shape: (batch_size, channels, n_mics, n_srcs, time)
    """
    # Create a tensor to hold all RIRs
    rir_tensor = torch.zeros(1, 1, n_mics, n_srcs, max_length)
    
    for mic_idx in range(n_mics):
        for src_idx in range(n_srcs):
            rir = IR[mic_idx][src_idx]
            # Truncate or zero-pad to max_length
            if len(rir) > max_length:
                rir = rir[:max_length]
            else:
                rir = np.pad(rir, (0, max_length - len(rir)))
            
            rir_tensor[0, 0, mic_idx, src_idx, :] = torch.tensor(rir)
    
    return rir_tensor


class ILZ_CNN_RIR(nn.Module):
    def __init__(self, M, S, L, T, time_length=512):
        super(ILZ_CNN_RIR, self).__init__()
        self.M, self.S, self.L, self.T = M, S, L, T
        self.time_length = time_length

        # 3D convolutions to process time dimension
        # Use smaller kernels and no stride to preserve information
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(1, 1, 16), padding=(0, 0, 8))
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(1, 1, 8), padding=(0, 0, 4))
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(1, 1, 4), padding=(0, 0, 2))
        
        # Global average pooling instead of calculating size
        self.pool = nn.AdaptiveAvgPool3d((M, S, 1))
        
        self.fc1 = nn.Linear(64 * M * S, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, S * T)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: (batch, 1, M, S, time_length)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.pool(x)  # (batch, 64, M, S, 1)
        x = x.view(x.size(0), -1)  # Flatten: (batch, 64 * M * S)
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))
        x = self.fc3(x)
        q = x.view(-1, self.S, self.T)  # (batch, S, T)
        return q

# Initialize model
n_mics = mic_positions.shape[1]

model = ILZ_CNN_RIR(n_mics, n_srcs, len(IR[0][0]), J, time_length=J)

try:
    model_path = os.path.join(script_dir, "SZC_model_rir.pth")
    model.load_state_dict(torch.load(model_path))
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model file 'SZC_model_rir.pth' not found in the script directory. Proceeding without loading.")
except Exception as e:
    print(f"Error loading model: {e}. Proceeding without loading.")

# Prepare the input tensor
x_rir = prepare_rir_input(IR, n_mics, n_srcs, max_length=J)

# FIX: Add torch.no_grad() and detach() to prevent gradient computation
with torch.no_grad():
    q_opt = model(x_rir)[0].detach().cpu().numpy()  # Convert to numpy array

print(f"Filter coefficients shape: {q_opt.shape}")

# ----------------------
# Pressure field visualization
# ----------------------
def pressure_field_2d(room_dim, sources, q_opt, lyd_data, grid_res=50, z_plane=1.5, J=512, fs=16000):
    """
    Compute and plot a 2D grid of sound pressure (SPL) in the room at a fixed z-plane.
    """
    L = len(sources)

    # Ensure q_opt is numpy array of shape (L, J)
    if q_opt.shape != (L, J):
        if q_opt.ndim == 3:  # (batch, L, J)
            q_opt = q_opt[0]
        q_opt = q_opt.reshape(L, J)
    
    q_matrix = q_opt.reshape(L, J)

    # 2D grid at fixed z
    x = np.linspace(0, room_dim[0], grid_res)
    y = np.linspace(0, room_dim[1], grid_res)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.full_like(X, z_plane)

    pressure_field = np.zeros_like(X, dtype=float)
    test_signal = np.array(lyd_data[:fs//2])  # Use short segment for speed

    for i in range(grid_res):
        for j in range(grid_res):
            point = [X[i, j], Y[i, j], Z[i, j]]
            h_point = []
            for src_idx, src_pos in enumerate(sources):
                r = np.linalg.norm(np.array(point) - np.array(src_pos))
                delay = int(r * fs / 343)
                h = np.zeros(J + 256)
                if delay < len(h):
                    h[delay] = 1 / (r + 1e-6)
                h_point.append(h)
            
            p = 0
            for l in range(L):
                filtered = lfilter(q_matrix[l], 1, test_signal)
                out_l = fftconvolve(filtered, h_point[l])
                p += np.sqrt(np.mean(out_l**2))  # RMS pressure
            pressure_field[i, j] = p

    # Convert to dB
    pressure_dB = 20 * np.log10(pressure_field / (np.max(pressure_field) + 1e-12) + 1e-12)

    # Define bright and dark zones
    bright_mask = X < room_dim[0] / 2
    dark_mask = X >= room_dim[0] / 2
    
    avg_bright = np.mean(pressure_field[bright_mask])
    avg_dark = np.mean(pressure_field[dark_mask])

    print(f"Average pressure in bright zone: {avg_bright:.4f}")
    print(f"Average pressure in dark zone: {avg_dark:.4f}")
    print(f"Pressure ratio (bright/dark) [dB]: {20 * np.log10(avg_bright / (avg_dark + 1e-12)):.2f} dB")

    # Plot
    plt.figure(figsize=(7, 7))
    im = plt.imshow(pressure_dB.T, origin='lower', extent=[0, room_dim[0], 0, room_dim[1]], 
                   aspect='auto', cmap='inferno')
    plt.colorbar(im, label='SPL [dB]')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Sound Pressure Level at z=1.5 m (Loaded Filter Coefficients)')
    
    # Plot speaker positions
    speaker_x = [s[0] for s in sources]
    speaker_y = [s[1] for s in sources]
    plt.scatter(speaker_x, speaker_y, c='cyan', marker='*', s=100, label='Speakers', edgecolors='black')
    
    # Add zone boundary
    plt.axvline(x=room_dim[0]/2, color='white', linestyle='--', alpha=0.7, label='Zone Boundary')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------
# Run the visualization
# ----------------------
pressure_field_2d(room_dim, sources, q_opt, lyd_data, grid_res=50, z_plane=1.5, J=J, fs=fs)

