import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
from scipy.signal import fftconvolve
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
import time, os
import Room_configuration as rc

# -------------------------
# Compute PM solution
# -------------------------
def PM_solution():
    '''
    print("Forming R_B, R_D and r_B...")
    tstart = time.perf_counter()
    R_B = U_B.T @ U_B      # (LJ x LJ)
    R_D = U_D.T @ U_D      # (LJ x LJ)
    r_B = U_B.T @ d_B      # (LJ,)
    '''
    R_B, R_D, r_B = rc.build_R()
    # normal eqn matrix
    A = R_B + rc.xi * R_D + rc.reg_eps * np.eye(R_B.shape[0])
    b = r_B  # note d_D=0 -> U_D.T d_D = 0
    print("Solving linear system A q = b ...")
    q_vec = np.linalg.solve(A, b)   # PM solution
    t_eig = time.perf_counter() - tstart
    print("Solved in {:.2f}s".format(t_eig))

    # reshape to per-source filters
    q_vec = np.real(q_vec)  # numerical safety
    q_vec = q_vec / (np.max(np.abs(q_vec)) + 1e-12)   # normalize for safe playback / plotting
    q_matrix = q_vec.reshape(rc.n_srcs, rc.J)
    print("q_matrix shape:", q_matrix.shape)
    return q_vec, q_matrix

# -------------------------
# Evaluate (same pressure_field routine as before)
# -------------------------
def pressure_field_from_q(q_matrix, IR, test_signal, room_dim, grid_res=50, z_plane=1.5):
    L = q_matrix.shape[0]
    x_grid = np.linspace(0, room_dim[0], grid_res)
    y_grid = np.linspace(0, room_dim[1], grid_res)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    Gx, Gy = X.shape
    pressure_field = np.zeros_like(X, dtype=float)
    for ix in range(Gx):
        for iy in range(Gy):
            point = np.array([X[ix,iy], Y[ix,iy], z_plane])
            p_sum = 0.0
            for l in range(L):
                drive = fftconvolve(test_signal, q_matrix[l])[:len(test_signal)]
                src_pos = np.array(rc.sources[l])
                r = np.linalg.norm(point - src_pos)
                delay = int(round(r * rc.fs_target / 343.0))
                h = np.zeros(max(1, delay+1))
                h[delay] = 1.0/(r + 1e-6)
                out = fftconvolve(drive, h)[:len(drive)]
                p_sum += np.sqrt(np.mean(out**2) + 1e-12)
            pressure_field[ix,iy] = p_sum
    return X, Y, pressure_field

if __name__ == "__main__":
    q_vec, q_matrix = PM_solution()
    print("Computing pressure field for visualization...")
    test_signal = rc.wav[:rc.fs_target//4] if len(rc.wav) >= rc.fs_target//4 else rc.wav
    tstart = time.perf_counter()
    Xg, Yg, P = pressure_field_from_q(q_matrix, rc.IR, test_signal, rc.room_dim, grid_res=rc.grid_res, z_plane=rc.z_plane)
    print("Computed in {:.2f}s".format(time.perf_counter() - tstart))

    bright_mask = Xg < (rc.room_dim[0]/2)
    dark_mask = ~bright_mask
    avg_bright = np.mean(P[bright_mask])
    avg_dark = np.mean(P[dark_mask])
    print(f"Average pressure (bright) = {avg_bright:.6f} ; (dark) = {avg_dark:.6f}")
    print("Contrast (bright/dark) [dB] =", 20.0 * np.log10((avg_bright + 1e-12) / (avg_dark + 1e-12)))

    P_db = 20.0 * np.log10(P / (np.max(P) + 1e-12) + 1e-12)
    plt.figure(figsize=(8,6))
    plt.imshow(P_db.T, origin='lower', extent=[0, rc.room_dim[0], 0, rc.room_dim[1]], cmap='inferno', aspect='auto')
    plt.colorbar(label='Relative dB')
    plt.scatter([s[0] for s in rc.sources], [s[1] for s in rc.sources], c='cyan', marker='*', s=100, edgecolors='k')
    plt.title('Relative SPL (dB) at z={:.2f} m (PM)'.format(rc.z_plane))
    plt.xlabel('x (m)'); plt.ylabel('y (m)')
    plt.tight_layout()
    plt.show()

    # Save q
    np.save("q_PM.npy", q_matrix)
    print("Saved q_PM.npy")

