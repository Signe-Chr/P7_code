import numpy as np
from scipy.linalg import toeplitz, eigh
import os
import time
import RIR_generator as RG
from tqdm import tqdm

dark_index = [0,1,2,3,4,5,6,7,8,9,10,11]
bright_index = [12]

J = 2
L = 1
M = 2

#x = np.array([1]+[0]*(3+6-2))


def load_ir(file_path):
    rir_dict = np.load(file_path, allow_pickle=True).item()  # Load dict from .npy
    return rir_dict

def toeplitz(x, n, K, J):
    X = np.zeros((K, J))

    for k in range(K):
        for j in range(J):
            index = n - k - j
            if 0 <= index < len(x):
                X[k, j] = x[index]
    return X


rir = np.array([[[1, 1]],
                [[1, 2]]], dtype = float)

x_inp = [1] + [0]*(2+2-3)

s = np.zeros([2, 2])
for i in range(len(x_inp)):
    s += toeplitz(x_inp, i, 2, 2)
print(s)




def R_c(x, rir):
    K = max(np.shape(rir))
    #mathcal_X = np.kron(toeplitz(x,0, K, J), np.eye(L))
    N = len(x)
    h = []
    for m in range(M):
        h_m = []
        for l in range(L):
            h_m += list(rir[m][l])
        h.append(h_m)
    h = np.array(h).T

    #print("h", h)
    #print("h[0]", h[0])
    zero_countnt = 0
    for n in range(N):
        H_B_n = np.matmul(toeplitz(x, n, K, J), h[0]).reshape(2,1)
        #print(np.shape(toeplitz(x, n, K, J)), np.shape(h[0]))
        #print("H_B", H_B_n)
        temp = np.matmul(H_B_n, H_B_n.T)
        if not np.all(temp==0):
            zero_countnt +=1
        print("temp", temp)
        if n == 0:
            R_B = np.zeros_like(temp)
        R_B += temp
    R_B = 1/(1*zero_countnt) * R_B
    print("R_b", R_B)

    zero_countnt = 0
    for n in range(N):
        H_D_n = np.matmul(toeplitz(x, n, K, J), h[1])
        #print("h[1]", h[1])
        #print("H_D_n", H_D_n, np.shape(H_D_n))
        temp = np.matmul(H_D_n.T, H_D_n)
        if not np.all(temp==0):
            zero_countnt +=1
        #print("temp", temp)
        if n == 0:
            R_D = np.zeros_like(temp)
        R_D += temp
    R_D = 1/(1*zero_countnt) * R_D

    return R_B, R_D


def acc_coeffs(R):
    lambda_vals, eigenvecs = eigh(R[0], R[1]+np.eye(2))
    print(R[0], R[1]+ np.eye(2))
    return eigenvecs, lambda_vals
print(acc_coeffs(R_c(x_inp, rir)))