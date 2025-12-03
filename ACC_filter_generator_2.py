import os
import time
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from scipy.linalg import toeplitz, eigh
from Test_train_split import J, L, indeces_bright, indeces_dark


M = len(indeces_dark)+len(indeces_bright)

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
    N = len(x)
    h = []
    for m in range(M):
        h_m = []
        for l in range(L):
            h_m += list(rir[m][l])
        h.append(h_m)
    h = np.array(h).T
    zero_countnt = 0

    for n in range(N):
        toe = toeplitz(x, n, K, J)
        H_B_n = np.matmul(np.kron(toe, np.eye(L)).T, h[:,indeces_bright])
        H_D_n = np.matmul(np.kron(toe, np.eye(L)).T, h[:,indeces_dark])

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

def load_save(x_input, opd):
    files = os.listdir("Data Archive")
    files.sort()
    print(files[:5])
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


if __name__ == "__main__":
    opdeling = [i for i in range(432)]
    op_1 = opdeling[:18]
    op_2 = opdeling[18: 36]
    op_3 = opdeling[36: 54]
    op_4 = opdeling[54: 72]
    op_5 = opdeling[72: 90]
    op_6 = opdeling[90: 108]
    op_7 = opdeling[108: 126]
    op_8 = opdeling[126: 144]
    op_9 = opdeling[144: 162]
    op_10 = opdeling[162: 180]
    op_11 = opdeling[180: 198]
    op_12 = opdeling[198: 216]
    op_13 = opdeling[216: 234]
    op_14 = opdeling[234: 252]
    op_15 = opdeling[252: 270]
    op_16 = opdeling[270: 288]
    op_17 = opdeling[288: 306]
    op_18 = opdeling[306: 324]
    op_19 = opdeling[324: 342]
    op_20 = opdeling[342: 360]
    op_21 = opdeling[360: 378]
    op_22 = opdeling[378: 396]
    op_23 = opdeling[396: 414]
    op_24 = opdeling[414: ]
    

    #load_save(load_wav_file())
    load_save(x_inp, <___>)

    
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
