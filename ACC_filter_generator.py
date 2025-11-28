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
    return rir_dict, file_path


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
    #mathcal_X = np.kron(toeplitz(x,0, K, J), np.eye(L))

    h = []
    for m in range(M):
        h_m = []
        for l in range(L):
            h_m += list(rir[m][l])
        h.append(h_m)
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
        dict, file = load_ir(r"Data_archive", 0)
        ir = dict["IR"]
        dict.update({"q_acc": acc_coeffs(R_c(input, ir, 1))})
        np.save(file, dict, allow_pickle=True)
        print(f"Saved filter {i} in {file}")
load_save(1)

dict, file = load_ir(r"Data_archive", 0)
