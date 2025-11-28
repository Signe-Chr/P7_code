import numpy as np
from scipy.linalg import toeplitz, eigh
import os
import time
import RIR_generator as RG


dark_index = [0,1,2,3,4,5,6,7,8,9,10,11]
bright_index = [12]

J = RG.J
L = 3
M = len(dark_index)+len(bright_index)


def load_ir(file_path):
    rir_dict = np.load(file_path, allow_pickle=True).item()  # Load dict from .npy
    return rir_dict

def toeplitz(x, n, K, J):
    X = np.zeros((K, J))

    for k in range(K):
        for j in range(J):
            index = n - k + j - 2
            if 0 <= index < len(x):
                X[k, j] = x[index]
    return X


def R_c(x, rir):
    K = max(np.shape(rir))
    #mathcal_X = np.kron(toeplitz(x,0, K, J), np.eye(L))
    N = RG.N
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

def load_save():
    for i in os.listdir("Data Archive"):
        dict = load_ir(f"Data Archive/{i}")
        ir = dict["IR"]
        dict.update({"q_acc": acc_coeffs(R_c(RG.x_input, ir))})
        np.save(f"Data Archive/{i}", dict, allow_pickle=True)
        print(f"Saved filter {i}")
load_save()

#dict, file = load_ir(r"Data Archive", 0)
