import os
import time
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from scipy.signal import resample_poly
from scipy.linalg import toeplitz, eigh
from Test_train_split import J, L, indeces_bright, indeces_dark




def load_wav_file():
    #wav_path = "relaxing-guitar-loop-v5-245859.wav"
    wav_path = "president-is-moron.wav"
    fs_wav, wav = wavfile.read(wav_path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    #wav = wav[int(5*fs_wav) : int(5.5*fs_wav)]
    wav = resample_poly(wav, up=1, down=3)[:int(1*16000)]
    #wavfile.write("president-is-moron_downsampled.wav", 16000, wav.astype(np.int16))
    wav = wav / np.max(np.abs(wav))  # scale to [-1,1]
    #x_input = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    #x_input = torchaudio.functional.resample(x_input, orig_freq=fs_wav, new_freq=16000)
    return wav.astype(np.float32)

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
    N = len(x)
    h = []
    for m in range(M):
        h_m = []
        for l in range(L):
            h_m += list(rir[m][l])
        h.append(h_m)
    h = np.array(h).T
    zero_countnt = 0

    I = np.eye(L)
    for n in tqdm(range(N)):
        toe = toeplitz(x, n, K, J)
        KRON = np.kron(toe, I).T
        H_B_n = np.matmul(KRON, h[:,indeces_bright])
        H_D_n = np.matmul(KRON, h[:,indeces_dark])

        temp_B = np.matmul(H_B_n, H_B_n.T)
        temp_D = np.matmul(H_D_n, H_D_n.T)
        if not np.all(toe==0):
            zero_countnt +=1
        if n == 0:
            R_B = np.zeros_like(temp_B)
            R_D = np.zeros_like(temp_D)
        R_B += temp_B
        R_D += temp_D
    R_B = 1/(len(indeces_bright)*zero_countnt) * R_B
    R_D = 1/(len(indeces_dark)*zero_countnt) * R_D

    return R_B, R_D

def acc_coeffs(R):
    lambda_vals, eigenvecs = eigh(R[0], R[1]+1e-6*np.eye(len(R[1])))
    #print(eigenvecs[:, -1])
    return eigenvecs[:, -1].reshape(3, 1024)

def load_save2(x_input, opd):
    files = os.listdir("Data Archive")
    files.sort()
    print(files[:])
    print(opd)
    times = []
    os.makedirs("Data Archive NEW", exist_ok=True)
    t = time.perf_counter()
    for u, i in tqdm(enumerate(files)):
        if u in opd:
            dict = load_ir(f"Data Archive/{i}")
            ir = dict["IR"]
            dict.update({"q_acc": acc_coeffs(R_c(x_input, ir))})
            np.save(f"Data Archive NEW/{i}", dict, allow_pickle=True)
            print(f"Saved filter {i}")
            t_new = time.perf_counter()
            times.append(t_new-t)
            t = t_new
    print(f"Total runtime: {np.sum(times):.2f}s\nAverage runtime: {np.mean(times):.2f}")

def load_save(x_input):
    all_files = os.listdir("Data Archive")
    all_files.sort()
    start = int(input("Start index (inc.): "))
    stop = int(input("Stop index (exc.): "))
    files = all_files[start:stop]
    times = []
    os.makedirs("Data Archive Speech", exist_ok=True)
    t = time.perf_counter()
    for file in tqdm(files):
        dict = load_ir(f"Data Archive/{file}")
        rir = dict["IR"]
        dict.update({"q_acc": acc_coeffs(R_c(x_input, rir))})
        np.save(f"Data Archive Speech/{file}", dict, allow_pickle=True)
        print(f"Saved filter {file}")
        t_new = time.perf_counter()
        times.append(t_new-t)
        t = t_new
    print(f"Total runtime: {np.sum(times):.2f}s\nAverage runtime: {np.mean(times):.2f}")

M = len(indeces_dark)+len(indeces_bright)

#x_inp = [1] + [0]*(J+512-2)

if __name__ == "__main__":
    x_inp = load_wav_file()
    load_save(x_inp)
    
    """file_path = "Data Archive/RIR_0_0_0_0_0.npy"
    #file_path = "Data Archive/RIR_0_0_0_0_1.npy"

    data = np.load(file_path, allow_pickle=True).item()

    if "q_acc" in data:
        q_acc = data["q_acc"]
        print(f"Filter coefficients (shape {q_acc.shape}):")
        non_zero_indices = np.argwhere(q_acc != 0)
        for idx in non_zero_indices:
            print(f"Index {tuple(idx)}: {q_acc[tuple(idx)]}")
    else:
        print("Key 'q_acc' not found in the loaded file.")"""
