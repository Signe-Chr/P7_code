import os, torch, torchaudio
import numpy as np
from scipy.io import wavfile
from Test_train_split import J, L, indeces_bright, indeces_dark
from scipy.linalg import toeplitz, eigh
from tqdm import tqdm

def load_wav_file():
    wav_path = "relaxing-guitar-loop-v5-245859.wav"
    fs_wav, wav = wavfile.read(wav_path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav = wav[5*fs_wav : 7*fs_wav]
    wav = wav / np.max(np.abs(wav))  # scale to [-1,1]
    x_input = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    x_input = torchaudio.functional.resample(x_input, orig_freq=fs_wav, new_freq=16000)
    return x_input

def load_ir(file_path):
    rir_dict = np.load(file_path, allow_pickle=True).item()  # Load dict from .npy
    return rir_dict

def toeplitz(x, n, K, J):
    X = np.zeros((K, J))

    for k in range(K):
        for j in range(J):
            index = n - k - j + 2
            if 0 <= index < len(x):
                X[k, j] = x[index]
    return X


#print(toeplitz([1], 0, 2, 2 ))
#print(toeplitz_fast([1], 0, 2, 2 ))

def R_c(x, rir):
    K = max(np.shape(rir))
    h = []
    N = len(x)
    print("Går i gang med m-loop")
    for m in range(M):
        h_m = []
        for l in range(L):
            h_m += list(rir[m][l])
        h.append(h_m)
    h = np.array(h).T

    print("Går i gang med n-loop nr 1")
    for n in tqdm(range(N)):
        H_B_n = np.matmul(np.kron(toeplitz(x, n, K, J), np.eye(L)).T, h[:,indeces_bright])
        temp = np.matmul(H_B_n, H_B_n.T)
        if n == 0:
            R_B = np.zeros_like(temp)
        R_B += temp
    R_B = 1/(len(indeces_bright)*N) * R_B

    print("Går i gang med n-loop nr 2")
    for n in range(N):
        H_D_n = np.matmul(np.kron(toeplitz(x, n, K, J), np.eye(L)).T, h[:,indeces_dark])
        temp = np.matmul(H_D_n, H_D_n.T)
        if n == 0:
            R_D = np.zeros_like(temp)
        R_D += temp
    R_D = 1/(len(indeces_dark)*N) * R_D

    return R_B, R_D

def acc_coeffs(R):
    lambda_vals, eigenvecs = eigh(R[0], R[1]+1e-6*np.eye(len(R[1])))
    return eigenvecs[:, -1].reshape(3, 1024)

def load_save(data_dir, save_dir, x_input):
    for i in tqdm(os.listdir(data_dir)):
        dict = load_ir(f"{data_dir}/{i}")
        print(" Dict loaded")
        ir = dict["IR"]
        dict.update({"q_acc": acc_coeffs(R_c(x_input, ir))})
        np.save(f"{save_dir}/{i}", dict, allow_pickle=True)
        #print(f"Saved filter {i}")

M = len(indeces_dark)+len(indeces_bright)
data_dir = "Data Archive"
save_dir = "Data Archive Music"
x_input = load_wav_file().squeeze(0)

load_save(data_dir, save_dir, x_input)



