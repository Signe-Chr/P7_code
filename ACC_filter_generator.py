import numpy as np
from scipy.linalg import toeplitz, eigh
import os
import time



input = [0,0,0,0,0,0,0,1,0,0]+[0 for i in range(100)]
J = 1024
L = 3
M = 13
dark_index = [0,1,2,3,4,5,6,7,8,9,10,11]
bright_index = [12]

def load_ir(folder_path, n):

    # List all files in folder and filter for .npy files starting with 'RIR_'
    files = [f for f in os.listdir(folder_path) if f.startswith("RIR_") and f.endswith(".npy")]

    if not files:
        raise FileNotFoundError("No RIR .npy files found in the specified folder.")

    # Sort files to ensure consistent ordering
    files.sort()

    if n < 0 or n >= len(files):
        raise IndexError(f"Requested index {n} is out of range. {len(files)} files available.")

    file_path = os.path.join(folder_path, files[n])
    rir_dict = np.load(file_path, allow_pickle=True).item()  # Load dict from .npy
    return rir_dict, files[n]



def toeplitz(x, n, K, J):
    X = np.zeros((K, J))

    for k in range(K):
        for j in range(J):
            index = n - k + j - 2
            if 0 <= index < len(x):
                X[k, j] = x[index]
    return X


def R_c(x, rir, N):
    K = max(np.shape(rir))
    mathcal_X = np.kron(toeplitz(x,0, K, J), np.eye(L))

    h = []
    for m in range(M):
        h_m = []
        for l in range(L):
            h_m += list(rir[m][l])
        h.append(np.array(h_m))
    h = np.array(h).T

    for n in range(N):
        H_B_n = np.matmul(np.kron(toeplitz(x, n, K, J), np.eye(L)).T, h[:,bright_index])
        temp = np.matmul(H_B_n, H_B_n.T)
        if n == 0:
            R_B = np.zeros_like(temp)
        R_B += temp
    R_B = 1/(len(bright_index)*N) * R_B

    for n in range(N):
        H_D_n = np.matmul(np.kron(toeplitz(x, n, K, J), np.eye(L)).T, h[:,dark_index])
        temp = np.matmul(H_D_n, H_D_n.T)
        if n == 0:
            R_D = np.zeros_like(temp)
        R_D += temp
    R_D = 1/(len(dark_index)*N) * R_D

    return R_B, R_D

def acc_coeffs(R):
    lambda_vals, eigenvecs = eigh(R[0], R[1]+1e-6*np.eye(len(R[1])))
    return eigenvecs[:, -1].reshape(3, 1024)

def load_save(N):
    for i in range(N):
        dict, file = load_ir(r"RIR_archive", 0)
        ir = dict["IR"]
        dict["q_matrix"] = acc_coeffs(R_c(input, ir, 1))
        np.save(file, dict, allow_pickle=True)
        print(f"Saved filter {i} in {file}")
load_save(1)

dict, file = load_ir(r"RIR_archive", 0)
print(dict["q_matrix"])


#print(np.shape(acc_coeffs(R_c(input, ir_test, 1))))




exit()

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

exit()
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
