import numpy as np
from scipy.linalg import toeplitz, eigh
import os
import time
import RIR_generator as RG
from tqdm import tqdm
import os, torch, torchaudio
from scipy.io import wavfile
from Test_train_split import J, L, indeces_bright, indeces_dark
from tqdm import tqdm

dark_index = [0,1,2,3,4,5,6,7,8,9,10,11]
bright_index = [12]

J = RG.J
L = 3
M = len(dark_index)+len(bright_index)

x_inp = [1] + [0]*(J+512-2)

def load_wav_file():
    wav_path = "relaxing-guitar-loop-v5-245859.wav"
    fs_wav, wav = wavfile.read(wav_path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav = wav[int(5*fs_wav) : int(5.5*fs_wav)]
    wav = wav / np.max(np.abs(wav))  # scale to [-1,1]
    #x_input = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    #x_input = torchaudio.functional.resample(x_input, orig_freq=fs_wav, new_freq=16000)
    return wav.astype(np.float32).flatten()

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

    for n in range(N):
        print(n/N)
        H_B_n = np.matmul(np.kron(toeplitz(x, n, K, J), np.eye(L)).T, h[:,bright_index])
        #print("toe", toeplitz(x, n, K, J))
        #print(H_B_n)
        temp = np.matmul(H_B_n, H_B_n.T)
        if n == 0:
            R_B = np.zeros_like(temp)
        R_B += temp
    R_B = 1/(len(bright_index)*N) * R_B

    for n in range(N):
        print(n/N)
        H_D_n = np.matmul(np.kron(toeplitz(x, n, K, J), np.eye(L)).T, h[:,dark_index])
        temp = np.matmul(H_D_n, H_D_n.T)
        if n == 0:
            R_D = np.zeros_like(temp)
        R_D += temp
    R_D = 1/(len(dark_index)*N) * R_D
    #print(R_D, R_D)

    return R_B, R_D

def acc_coeffs(R):
    lambda_vals, eigenvecs = eigh(R[0], R[1]+1e-6*np.eye(len(R[1])))
    #print(eigenvecs[:, -1])
    return eigenvecs[:, -1].reshape(3, 1024)

def load_save(x_input):
    for u, i in tqdm(enumerate(os.listdir("Data Archive"))):
        if u == 0:
            dict = load_ir(f"Data Archive/{i}")
            ir = dict["IR"]
            dict.update({"q_acc": acc_coeffs(R_c(x_input, ir))})
            np.save(f"Data Archive/{i}", dict, allow_pickle=True)
            print(f"Saved filter {i}")


if __name__ == "__main__":
    #load_save(load_wav_file())
    load_save(x_inp)
    
    file_path = "Data Archive/RIR_0_0_0_0_0.npy"
    #file_path = "Data Archive/RIR_0_0_0_0_1.npy"

    data = np.load(file_path, allow_pickle=True).item()

    if "q_acc" in data:
        q_acc = data["q_acc"]
        print(f"Filter coefficients (shape {q_acc.shape}):")
        non_zero_indices = np.argwhere(q_acc != 0)
        for idx in non_zero_indices:
            print(f"Index {tuple(idx)}: {q_acc[tuple(idx)]}")
    else:
        print("Key 'q_acc' not found in the loaded file.")
