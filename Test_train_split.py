import os, torch, torchaudio
import numpy as np
from sklearn.model_selection import train_test_split
from Dataset_class import CustomDataset
from torch.utils.data import DataLoader
from scipy.io import wavfile


L = 3
J = 1024
indeces_dark = [0,1,2,3,4,5,6,7,8,9,10,11]
indeces_bright = [12]
x_input_kronecker = [1] + [0]*(J+512-2)

def load_test_train_data(test_size=0.25, random_seed=42):
    data_dir = "Data Archive"
    full_data = os.listdir(data_dir)

    # Perform train/test split with fixed random seed
    files_train, files_test = train_test_split(
        full_data, test_size=test_size, random_state=random_seed, shuffle=True
    )

    # Create dataset instances
    temp_var_train = CustomDataset(data_dir, files_train)
    temp_var_test = CustomDataset(data_dir, files_test)

    train_loader = DataLoader(temp_var_train, batch_size=len(temp_var_train), shuffle=False)
    test_loader = DataLoader(temp_var_test, batch_size=len(temp_var_test), shuffle=False)
    data_train = [batch for batch in train_loader][0]
    data_test = [batch for batch in test_loader][0]

    return data_test, data_train

def load_wav_file():
    wav_path = "relaxing-guitar-loop-v5-245859.wav"
    fs_wav, wav = wavfile.read(wav_path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav = wav[5*fs_wav : 7*fs_wav]
    wav = wav / np.max(np.abs(wav))  # scale to [-1,1]
    x_input = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    x_input = torchaudio.functional.resample(x_input, orig_freq=fs_wav, new_freq=16000)
    x_input_wav = x_input.squeeze(0)
    return x_input_wav