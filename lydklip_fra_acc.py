
import torch
import torch.nn as nn


import numpy as np

def load_dict_from_npy(path):
    return np.load(path, allow_pickle=True).item()

d = load_dict_from_npy(r"C:\Users\marst\OneDrive\Skrivebord\UNI\S. 7\PROJEKT\P7\Data Archive\RIR_1_1_1_1_1.npy")

q = d["q_acc"][0]
h = d["IR"][12][0]


from scipy.io import wavfile

sample_rate, audio = wavfile.read(r"C:\Users\marst\OneDrive\Skrivebord\UNI\S. 7\PROJEKT\P7\president-is-moron.wav")


if audio.ndim == 2:
    audio = audio.mean(axis=1)


#print(np.shape(audio))


import numpy as np
import soundfile as sf

def convolve_and_save_wav(list1, list2, list3, sr, out_path):
    """
    Convolve 3 lists together and save as a mono WAV file.

    Parameters:
        list1, list2, list3 : arrays/lists of numbers
        sr                  : sample rate (e.g., 44100)
        out_path            : path to output WAV file
    """

    # Convert input to numpy arrays
    a = np.array(list1, dtype=float)
    b = np.array(list2, dtype=float)
    c = np.array(list3, dtype=float)

    # Step 1: convolve first two
    conv_ab = np.convolve(a, b, mode="full")

    # Step 2: convolve result with third
    conv_abc = np.convolve(conv_ab, c, mode="full")

    conv_abc = conv_abc

    
    max_val = np.max(np.abs(conv_abc))
    if max_val > 0:
        conv_abc = conv_abc / max_val

    print(max(conv_abc), min(conv_abc))
    # Save WAV (mono)
    sf.write(out_path, conv_abc.astype(np.float32), sr)

    print(f"Saved WAV to: {out_path}")
    return conv_abc

convolve_and_save_wav(q,h,audio, 44100, r"C:\Users\marst\OneDrive\Skrivebord\UNI\S. 7\PROJEKT\P7\acc_moron_b.wav")