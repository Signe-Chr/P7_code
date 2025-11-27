import numpy as np
from scipy.signal import fftconvolve
from scipy.linalg import eigh
import time
import os
import Room_configuration as rc  # Make sure this points to your module

def ACC_solution():
    """
    Solve generalized eigenproblem R_B q = gamma R_D q
    Returns the eigenvector q_vec and reshaped q_matrix per source.
    """
    R_B, R_D_reg, r_d = rc.build_R()
    n_srcs = len(rc.sources)

    print("Solving generalized eigenvalue problem (may take a while)...")
    tstart = time.perf_counter()
    eigvals, eigvecs = eigh(R_B, R_D_reg)   # ascending eigenvalues
    t_eig = time.perf_counter() - tstart
    print(f"Eigenproblem solved in {t_eig:.2f}s; number of eigvals = {len(eigvals)}")

    # Pick eigenvector with largest eigenvalue
    idx = np.argmax(eigvals)
    gamma_max = eigvals[idx]
    q_vec = np.real(eigvecs[:, idx])
    print("Maximum Acoustic Contrast (gamma_max) =", gamma_max)

    # Normalize for sensible amplitude
    q_vec = q_vec / (np.max(np.abs(q_vec)) + 1e-12)

    # Reshape into per-source filters: (n_srcs, J)
    q_matrix = q_vec.reshape(n_srcs, rc.J)
    print("q_matrix shape:", q_matrix.shape)

    return q_vec, q_matrix

if __name__ == "__main__":
    save_dir = "q_matrices"
    os.makedirs(save_dir, exist_ok=True)

    # Loop over all RIR configurations in your archive
    rir_files = [f for f in os.listdir("RIR_archive") if f.endswith(".npy")]  # or .wav/.npz depending on format

    for rir_file in rir_files:
        config_name = os.path.splitext(rir_file)[0]
        print(f"Processing configuration: {config_name}")

        # Load RIR or other room data
        rc.load_RIR(os.path.join("RIR_archive", rir_file))  # Make sure you have a function to load it into rc.IR etc.

        # Compute ACC filters
        q_vec, q_matrix = ACC_solution()

        # Save q_matrix for this configuration
        out_path = os.path.join(save_dir, f"{config_name}_q.npy")
        np.save(out_path, q_matrix)
        print(f"Saved q_matrix to {out_path}")
