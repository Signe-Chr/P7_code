import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchsummary import summary
import Dataset_generator_script as dgs
from tqdm import tqdm
from Dataset_class import CustomDataset, L, J
from multiprocessing import cpu_count


def compute_H_matrix(rir_array, fs=16000, n_fft=None):
    """
    Compute the frequency-domain transfer matrix H[k]
    from a set of impulse responses.

    Parameters
    ----------
    rir_array : np.ndarray, shape (n_mics, n_srcs, n_samples)
        Time-domain impulse responses for each mic–source pair.
        rir_array[m, s, :] = impulse response from source s to mic m.
    fs : int, optional
        Sampling frequency in Hz (default: 16000).
    n_fft : int, optional
        FFT length. If None, uses next power of 2 above rir length.

    Returns
    -------
    H : np.ndarray, shape (n_mics, n_srcs, n_freqs)
        Frequency response matrix for all microphone–source pairs.
    freqs : np.ndarray
        Frequency vector (in Hz) for the frequency bins.
    """
    # --- Input validation ---
    if rir_array.ndim != 3:
        raise ValueError(f"Expected rir_array of shape (n_mics, n_srcs, n_samples), got {rir_array.shape}")

    n_mics, n_srcs, n_samples = rir_array.shape

    # --- Choose FFT length ---
    if n_fft is None:
        n_fft = 2 ** int(np.ceil(np.log2(n_samples)))  # next power of 2

    n_freqs = n_fft // 2 + 1

    # --- Allocate frequency-domain matrix ---
    H = np.zeros((n_mics, n_srcs, n_freqs), dtype=np.complex128)

    # --- Compute FFT for each mic–source pair ---
    for m in range(n_mics):
        for s in range(n_srcs):
            h = rir_array[m, s, :]
            H[m, s, :] = np.fft.rfft(h, n=n_fft)

    # --- Frequency axis ---
    freqs = np.fft.rfftfreq(n_fft, 1 / fs)

    return H, freqs

def L_2_loss(test_filter_flat: torch.Tensor, candidate_filter_flat: torch.Tensor):
    """Cosine distance between two flattened filters."""
    y_test_norm = F.normalize(test_filter_flat, p=2, dim=1)
    y_cand_norm = F.normalize(candidate_filter_flat, p=2, dim=1)
    similarity = torch.mm(y_test_norm, y_cand_norm.T)
    cosine_distance = 1 - similarity.squeeze()
    return cosine_distance

