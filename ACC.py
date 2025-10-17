import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
from scipy.signal import lfilter, fftconvolve
from scipy.linalg import toeplitz, eigh
import matplotlib.pyplot as plt
import time
import os
import Room_configuration as rc


def ACC_solution():
    # Solve generalized eigenproblem R_B q = gamma R_D q
    # The solution q is the eigenvector corresponding to the maximum eigenvalue (gamma)
    R_B, R_D_reg, r_d = rc.build_R()
    n_srcs = len(rc.sources)

    print("Solving generalized eigenvalue problem (this may take a while for large LJ)...")
    tstart = time.perf_counter()
    eigvals, eigvecs = eigh(R_B, R_D_reg)   # returns ascending eigenvalues
    t_eig = time.perf_counter() - tstart
    print(f"Eigenproblem solved in {t_eig:.2f} s; number of eigvals = {len(eigvals)}")

    # pick eigenvector with largest eigenvalue (highest acoustic contrast)
    idx = np.argmax(eigvals)
    gamma_max = eigvals[idx]
    q_vec = eigvecs[:, idx]
    print("Maximum Acoustic Contrast (gamma_max) =", gamma_max)

    # q_vec may be complex if numerical issues; ensure real
    q_vec = np.real(q_vec)

    # scale / normalize q_vec to a sensible speaker amplitude level before plotting/simulation
    # Normalization: max absolute coefficient of the concatenated filter vector is 1
    q_vec = q_vec / (np.max(np.abs(q_vec)) + 1e-12)

    # reshape q into per-source filters: (n_srcs, J)
    q_matrix = q_vec.reshape(n_srcs, rc.J)

    print("q_matrix shape:", q_matrix.shape)
    return(q_vec, q_matrix)

# -------------------------
# Compute resulting pressure field (uses direct path approximation for visualization)
# -------------------------
def pressure_field_from_q(q_matrix, IR, test_signal, sources, room_dim, fs_target, grid_res=50, z_plane=1.5):
    """
    Compute RMS pressure field (in pressure units, not dB) on a 2D grid at z=z_plane.
    NOTE: This uses the simple direct-path model for the visualization grid, NOT the full RIRs.
    """
    L = q_matrix.shape[0]
    x_grid = np.linspace(0, room_dim[0], grid_res)
    y_grid = np.linspace(0, room_dim[1], grid_res)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    Gx, Gy = X.shape
    pressure_field = np.zeros_like(X, dtype=float)
    
    speed_of_sound = 343.0

    # Iterate over every point on the visualization grid
    for ix in range(Gx):
        for iy in range(Gy):
            point = np.array([X[ix,iy], Y[ix,iy], z_plane])
            p_sum = 0.0
            
            # The RMS value at the point is approximated as the sum of RMS contributions 
            # from each loudspeaker (a simple, non-coherent approximation)
            for l in range(L):
                # 1. Compute loudspeaker drive signal (convolved with the filter q_l)
                drive = fftconvolve(test_signal, q_matrix[l])[:len(test_signal)]
                
                # 2. Approximate acoustic path (direct path only)
                src_pos = np.array(sources[l])
                r = np.linalg.norm(point - src_pos)
                delay = int(round(r * fs_target / speed_of_sound))
                
                # Simple direct-path weight (1/r)
                if r > 1e-6:
                    h = np.zeros(max(1, delay + 1))
                    h[delay] = 1.0 / r 
                else:
                    h = np.array([1.0]) # Handles edge case where point is exactly at source
                    
                # The total pressure at the grid point is the convolution of 
                # the drive signal with the path approximation h.
                out = fftconvolve(drive, h)[:len(drive)]
                
                # Use the RMS value of the resulting pressure waveform as the measure
                p_sum += np.sqrt(np.mean(out**2) + 1e-12)
                
            pressure_field[ix,iy] = p_sum
            
    return X, Y, pressure_field

if __name__ == "__main__":
    q_vec, q_matrix = ACC_solution()
    print("Computing pressure field (coarse grid for speed)...")
    # Use a short segment of the signal for visualization to speed up convolution
    test_signal = rc.wav[:rc.fs_target//4] if len(rc.wav) >= rc.fs_target//4 else rc.wav
    tstart = time.perf_counter()
    # Note: we pass the sources list and fs_target to the function for direct path calculation
    Xg, Yg, P = pressure_field_from_q(q_matrix, rc.IR, test_signal, rc.sources, rc.room_dim, rc.fs_target, grid_res=rc.grid_res, z_plane=rc.z_plane)
    print("Pressure computed in {:.2f}s".format(time.perf_counter() - tstart))

    # Compute averages for bright/dark masks
    bright_mask = Xg < (rc.room_dim[0]/2)
    dark_mask = ~bright_mask
    avg_bright = np.mean(P[bright_mask])
    avg_dark = np.mean(P[dark_mask])
    print(f"Average pressure (bright) = {avg_bright:.6f} ; (dark) = {avg_dark:.6f}")
    print("Contrast (bright/dark) [dB] =", 20.0 * np.log10((avg_bright + 1e-12) / (avg_dark + 1e-12)))

    # plot SPL-like plot (convert to relative dB)
    # Normalize to the maximum pressure on the grid for relative dB scale
    P_db = 20.0 * np.log10(P / (np.max(P) + 1e-12) + 1e-12)
    plt.figure(figsize=(8,6))
    plt.imshow(P_db.T, origin='lower', extent=[0, rc.room_dim[0], 0, rc.room_dim[1]], cmap='inferno', aspect='auto')
    plt.colorbar(label='Relative dB')
    plt.scatter([s[0] for s in rc.sources], [s[1] for s in rc.sources], c='cyan', marker='*', s=100, edgecolors='k')
    plt.title('Relative SPL (dB) at z={:.2f} m (Time-Domain MSIR)'.format(rc.z_plane))
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.tight_layout()
    plt.show()

    # Save q to file
    out_q_path = "q_solution_td.npy"
    np.save(out_q_path, q_matrix)
    print("Saved q_matrix to", out_q_path)
