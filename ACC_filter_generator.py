import numpy as np
from scipy.linalg import toeplitz, eigh
import os
import time

def compute_ACC_from_file(file_path):
    """
    Compute ACC filter from a single RIR archive file.
    Works when IR is a list of lists of arrays (IR[source][mic] = np.array)
    Safely skips mic indices that are out of range.
    """
    data = np.load(file_path, allow_pickle=True).item()
    IR = data['IR']
    sources = data['sources_position']
    J = data['J']
    bright_idx = data['bright_zone_mics_index']
    dark_idx = data['dark_zone_mics_index']

    n_srcs = len(IR)
    n_mics = [len(IR[i]) for i in range(n_srcs)]  # list of #mics per source
    N = IR[0][0].shape[0]  # length of IR for one mic (assume all same length)

    # Initialize matrices
    R_B = np.zeros((n_srcs*J, n_srcs*J))
    R_D = np.zeros((n_srcs*J, n_srcs*J))

    print(f"Building R_B and R_D for {os.path.basename(file_path)}...")
    tstart = time.perf_counter()

    for i in range(n_srcs):
        for j in range(n_srcs):
            # bright region
            for m in bright_idx:
                if m >= n_mics[i]:
                    continue  # skip if index out of range
                h = IR[i][m]
                T_B = toeplitz(np.r_[h, np.zeros(J-1)], np.zeros(J))
                R_B[i*J:(i+1)*J, j*J:(j+1)*J] += T_B.T @ T_B
            # dark region
            for m in dark_idx:
                if m >= n_mics[i]:
                    continue
                h = IR[i][m]
                T_D = toeplitz(np.r_[h, np.zeros(J-1)], np.zeros(J))
                R_D[i*J:(i+1)*J, j*J:(j+1)*J] += T_D.T @ T_D

    R_D_reg = R_D + 1e-6*np.eye(n_srcs*J)

    t_matrix = time.perf_counter() - tstart
    print(f"Matrices built in {t_matrix:.2f}s.")

    # Solve generalized eigenproblem
    print("Solving generalized eigenvalue problem...")
    tstart = time.perf_counter()
    eigvals, eigvecs = eigh(R_B, R_D_reg)
    t_eig = time.perf_counter() - tstart
    print(f"Eigenproblem solved in {t_eig:.2f}s; number of eigvals = {len(eigvals)}")

    idx = np.argmax(eigvals)
    q_vec = np.real(eigvecs[:, idx])
    q_vec /= np.max(np.abs(q_vec)) + 1e-12
    q_matrix = q_vec.reshape(n_srcs, J)
    print("q_matrix shape:", q_matrix.shape)

    return q_vec, q_matrix


if __name__ == "__main__":
    rir_folder = "RIR_archive"
    save_folder = "ACC_filter_archive"
    os.makedirs(save_folder, exist_ok=True)

    for rir_file in os.listdir(rir_folder):
        if not rir_file.endswith(".npy"):
            continue

        config_name = os.path.splitext(rir_file)[0]
        out_path = os.path.join(save_folder, f"{config_name}_q.npy")

        # Skip if already computed
        if os.path.exists(out_path):
            print(f"\nSkipping already processed configuration: {config_name}")
            continue

        print(f"\nProcessing configuration: {config_name}")
        file_path = os.path.join(rir_folder, rir_file)
        q_vec, q_matrix = compute_ACC_from_file(file_path)

        np.save(out_path, q_matrix)
        print(f"Saved q_matrix to {out_path}")
