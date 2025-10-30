import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, fftconvolve
# Assuming the file 'Dataset_generator_script.py' exists and is imported correctly
import Dataset_generator_script as dgs 

# ================================================================
# 1. Define the model (FIXED ARCHITECTURE)
# The architecture is updated to match the saved weights in mlp_weights.pth
# ================================================================
class SoftFilterNet(nn.Module):
    def __init__(self, input_size, num_filters, filter_dim, filters_tensor):
        super().__init__()
        # Layers reconstructed based on the state_dict errors:
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)       # Matched checkpoint size
        self.fc3 = nn.Linear(256, 512)       # Matched checkpoint size
        self.fc4 = nn.Linear(512, num_filters) # Matched checkpoint size (maps to num_filters)
        self.register_buffer("filters", filters_tensor)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))             # Added activation for the new fc3
        logits = self.fc4(x)                # Changed to fc4 as final layer
        weights = F.softmax(logits, dim=1)
        predicted_filters = weights @ self.filters
        return predicted_filters, weights


# ================================================================
# 2. Load the trained model + filter dictionary (FIXED DATA LOADING)
# ================================================================
data = np.load("VAST_filter_archive.npy", allow_pickle=True).item()
filters_list = []
for inner in data.values():
    q = inner.get("q_matrix", np.zeros((3, 1024), dtype=np.float32))
    # FIX: Use the actual q_matrix from the data archive, not an array of ones.
    filters_list.append(q.reshape(-1))# flatten
filters_np = np.stack(filters_list).astype(np.float32)
filters_tensor = torch.unique(torch.from_numpy(filters_np), dim=0)
num_filters, filter_dim = filters_tensor.shape

input_size = 6
model = SoftFilterNet(input_size, num_filters, filter_dim, filters_tensor)
model.load_state_dict(torch.load("mlp_weights.pth", map_location="cpu"))
model.eval()
print(f"\nLoaded SoftFilterNet model with weights from 'mlp_weights.pth'")
print(f"Model successfully loaded with {num_filters} unique filters.")

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
                out_l = fftconvolve(filtered, h)
                
                # Compute RMS pressure contribution
                p += np.sqrt(np.mean(out_l**2))
            
            pressure_field[i, j] = p

    # Normalize and convert to dB
    pressure_dB = 20 * np.log10(pressure_field / (np.max(pressure_field) + 1e-12))

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
    plt.title('Predicted Sound Pressure Field (SoftFilterNet)')

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


# ================================================================
# 4. Predict filter + visualize
# ================================================================

Center = np.array([2.0, 2.0, 1.5])
room_dim = [5, 4, 3]
sources_position_list = np.array([
    [Center[0]-0.1, Center[1]-0.2, Center[2]],
    [Center[0]+0.1, Center[1]-0.2, Center[2]],
    [Center[0],Center[1]-0.2, Center[2]+0.2],
])

rt60 = 0.4
tilt_rotation = np.deg2rad(15)
user_rotation = np.deg2rad(45)

# NOTE: The rotation logic below seems to be intended to calculate source positions
# relative to a rotated user, but it's not actually used in X_input.
# Since X_input only uses the raw Center coordinates and the rotations/tilt angles,
# the code will likely run now, but the geometry calculation might be redundant 
# for the model's prediction. We keep the original calculation for visualization purposes.

user_orientation = np.array([
                        [np.cos(user_rotation),-np.sin(user_rotation), 0],
                        [np.sin(user_rotation),np.cos(user_rotation), 0],
                        [0,0,1]
                    ])

rotation_x = np.array([
                              [1,0,0],
                              [0, np.cos(tilt_rotation),-np.sin(tilt_rotation)],
                              [0, np.sin(tilt_rotation),np.cos(tilt_rotation)]
                          ])

orientation_source_temp = np.matmul(user_orientation, sources_position_list.T - Center.reshape(-1, 1))

orientation_source_final = np.matmul(rotation_x, orientation_source_temp).T
orientation_source_final += Center

# Model Input: [rt60, phone_tilt, user_rotation, spatial_position_x, y, z]
X_input = np.array([[rt60, tilt_rotation, user_rotation] + list(Center)], dtype=np.float32)
X_tensor = torch.from_numpy(X_input)

with torch.no_grad():
    predicted_filter, weights = model(X_tensor)

# Note: predicted_filter[0] is used because the batch size is 1.
q_opt = predicted_filter[0].cpu().numpy().reshape(3, -1)
print(f"Predicted filter shape: {q_opt.shape}")

# Using the *original* source positions for visualization as the prediction is based on the unrotated center.
# If the prediction logic includes the rotations, the 'orientation_source_final' should be used.
# For now, we use the output of the rotation calculation for consistency with original script logic.
pressure_field_2d(room_dim, orientation_source_final, q_opt, Center, fs=16000, grid_res=40, J=q_opt.shape[1], r_zone=1.0)
