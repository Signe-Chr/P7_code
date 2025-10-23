import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
import time
import os
import torch
import torch.fft as tfft

# ---------------------------
# Helpers: device selection
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# Room setup (same as before)
# ---------------------------
def setup_acoustic_scenario(sources, mic_positions_list, bright_zone_mics_index,
                            dark_zone_mics_index, fs_target, room_dim, rt60,
                            mic_directions, user_rotation):
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(room_dim, fs=fs_target, materials=pra.Material(e_absorption), max_order=max_order)
    for s in sources:
        room.add_source(s)
    mic_positions = np.array(mic_positions_list).T
    mic_array = pra.MicrophoneArray(mic_positions, room.fs)
    room.add_microphone_array(mic_array)
    try:
        room.mic_array.set_directivity(
            mic_directions[:-1] + [pra.directivities.HyperCardioid(
                pra.directivities.DirectionVector(user_rotation - np.pi/2)
            )]
        )
    except Exception:
        pass
    print(f"Computing RIRs for {mic_positions.shape[1]} mics and {len(sources)} sources...")
    room.compute_rir()
    IR = room.rir
    return IR, len(bright_zone_mics_index), len(dark_zone_mics_index)

# ---------------------------
# Torch utilities for batched FFT conv
# ---------------------------
def _pad_ir_list(IR, mic_indices):
    """Collect IRs for chosen mics and sources, pad to common length, return numpy array shape (M_sel, L, h_len)"""
    chosen = [m for m in mic_indices if 0 <= m < len(IR)]
    if len(chosen) == 0:
        raise ValueError("No valid mic indices")
    L = len(IR[0])  # number of sources per mic
    # determine max length among chosen mics & sources
    max_h = 0
    for m in chosen:
        for l in range(L):
            max_h = max(max_h, len(IR[m][l]))
    # create padded array
    H = np.zeros((len(chosen), L, max_h), dtype=np.float32)
    for idx_m, m in enumerate(chosen):
        for l in range(L):
            h = np.asarray(IR[m][l], dtype=np.float32)
            H[idx_m, l, :len(h)] = h
    return H, chosen

def _batched_conv_x_h_torch(x, H_np, N, J, device=device):
    """
    x: 1D numpy array length >= N or torch tensor (1D)
    H_np: numpy array with shape (M_sel, L, h_len) float32
    returns: torch tensor U of shape (M_sel, L, out_len) where out_len = N+J-1
    Uses batched FFT convolution on device.
    """
    # convert x to torch and move to device
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    M_sel, L, h_len = H_np.shape
    out_len = N + J - 1
    nfft = int(2 ** np.ceil(np.log2(out_len + h_len)))  # safe nfft (can be smaller but fine)
    # rfft of x once
    Xf = tfft.rfft(x_t, n=nfft)
    # prepare H tensor on device
    H_t = torch.as_tensor(H_np, dtype=torch.float32, device=device)  # (M_sel, L, h_len)
    # compute rfft(H) for all (M_sel*L)
    Hf = tfft.rfft(H_t, n=nfft)  # (M_sel, L, nfft//2+1), complex
    # multiply in freq domain: broadcast Xf to (M_sel, L, freq)
    # Xf shape: (freq,), expand to (M_sel, L, freq)
    Xf_exp = Xf.unsqueeze(0).unsqueeze(0)  # (1,1,freq)
    Yf = Hf * Xf_exp  # (M_sel, L, freq)
    # inverse rfft to time domain
    y = tfft.irfft(Yf, n=nfft)  # (M_sel, L, nfft)
    y = y[..., :out_len]  # trim
    return y  # torch tensor on device (M_sel, L, out_len)

# ---------------------------
# GPU-accelerated R and r_B builders
# ---------------------------
def build_R_from_micset_torch(mic_indices, x, IR, N, J, n_srcs, reg_eps=0.0, device=device):
    """
    Fast GPU/CPU torch implementation:
    - pads IRs for chosen mics
    - computes batched conv u = x * h_{m,l}
    - forms sliding windows via unfold
    - computes R via einsum
    Returns R on CPU (numpy float32) for downstream eigh (scipy) which expects numpy.
    """
    H_np, chosen = _pad_ir_list(IR, mic_indices)  # (M_sel, L, h_len)
    M_sel = H_np.shape[0]
    # batched convolution on device
    U_all = _batched_conv_x_h_torch(x, H_np, N, J, device=device)  # (M_sel, L, out_len)
    # create sliding windows length J -> use torch.unfold over last dim
    # we want W shape (M_sel, N, L, J)
    # first reorder to (M_sel, L, out_len)
    U_all = U_all  # already this shape
    # unfold
    # reshape for unfold: merge M_sel and L dims -> (M_sel*L, out_len)
    ML = M_sel * n_srcs
    U_flat = U_all.view(ML, -1)  # (M_sel*L, out_len)
    W = U_flat.unfold(dimension=1, size=J, step=1)  # (ML, N, J)
    # reshape back to (M_sel, L, N, J)
    W = W.view(M_sel, n_srcs, N, J).permute(0, 2, 1, 3)  # (M_sel, N, L, J)
    # reshape to (M_sel, N, L*J)
    W_reshaped = W.reshape(M_sel, N, n_srcs * J)  # torch tensor (on device)
    # compute R = einsum over m and n: 'mna,mnb->ab'
    R_t = torch.einsum('mna,mnb->ab', W_reshaped, W_reshaped)
    # normalize to match original normalization (divide by M and N)
    R_t = R_t / float(M_sel * N)
    # regularize and symmetrize
    if reg_eps > 0:
        R_t = R_t + reg_eps * torch.eye(R_t.shape[0], dtype=R_t.dtype, device=device)
    R_t = 0.5 * (R_t + R_t.T)
    # move back to CPU numpy for scipy eigh
    return R_t.cpu().numpy()

def compute_rB_torch(bright_mics_index, x, IR, d_B, N, J, n_srcs, device=device):
    """
    Compute r_B using torch on GPU/CPU and return numpy array.
    d_B is numpy (N, M_b).
    """
    H_np, chosen = _pad_ir_list(IR, bright_mics_index)
    M_sel = H_np.shape[0]
    U_all = _batched_conv_x_h_torch(x, H_np, N, J, device=device)  # (M_sel, L, out_len)
    # unfold sliding windows as above
    ML = M_sel * n_srcs
    U_flat = U_all.view(ML, -1)
    W = U_flat.unfold(dimension=1, size=J, step=1)
    W = W.view(M_sel, n_srcs, N, J).permute(0, 2, 1, 3)  # (M_sel, N, L, J)
    W_reshaped = W.reshape(M_sel, N, n_srcs * J)  # (M_sel, N, LJ)
    # d_B selection: build d_sel (N, M_sel) assuming bright_mics_index order corresponds to columns in d_B
    # if d_B columns correspond exactly to bright_mics_index order, just select first M_sel cols
    d_sel = d_B[:, :M_sel]  # (N, M_sel)
    d_sel_t = torch.as_tensor(d_sel, dtype=torch.float32, device=device)  # (N, M_sel)
    # einsum: 'mna,nm->a' (note transposition of d_sel)
    r_t = torch.einsum('mna,nm->a', W_reshaped, d_sel_t, optimize=True)  # (LJ,)
    r_t = r_t / float(M_sel * N)
    return r_t.cpu().numpy()

# ---------------------------
# Main design function (torch)
# ---------------------------
def design_vast_filter(sources, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index,
                           wav_path, rt60, direction_list, user_rotation, fs_target, J, N, V, mu,
                           room_dim, reg_eps, target_amplitude, device=device):
    print("Starting GPU-accelerated VAST design...")
    t0 = time.perf_counter()

    if not os.path.exists(wav_path):
        raise FileNotFoundError(wav_path)
    fs_wav, wav = wavfile.read(wav_path)
    if fs_wav != fs_target:
        print("Warning: sample rate mismatch (no resample).")
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav[:, 0]
    wav /= (np.max(np.abs(wav)) + 1e-12)
    x = wav[:N] if len(wav) >= N else np.pad(wav, (0, N - len(wav)))

    n_srcs = len(sources)
    n_mics = len(mic_positions_list)

    IR, M_b, M_d = setup_acoustic_scenario(sources, mic_positions_list,
                                           bright_zone_mics_index, dark_zone_mics_index,
                                           fs_target, room_dim, rt60, direction_list, user_rotation)

    # Build R_B, R_D on GPU, then move to numpy for eigh
    t1 = time.perf_counter()
    R_B = build_R_from_micset_torch(bright_zone_mics_index, x, IR, N, J, n_srcs, reg_eps=0.0, device=device)
    R_D = build_R_from_micset_torch(dark_zone_mics_index, x, IR, N, J, n_srcs, reg_eps=reg_eps, device=device)
    print(f"Built R_B and R_D in {time.perf_counter() - t1:.2f} s")

    # Solve GEP
    print("Solving GEP (scipy eigh)...")
    lambda_vals, U = np.linalg.eig(np.linalg.pinv(R_D) @ R_B) if True else np.linalg.eig(R_B)  # fallback approach
    # fallback: prefer scipy.linalg.eigh(R_B, R_D) if available and R_D posdef
    try:
        from scipy.linalg import eigh as sp_eigh
        lambda_vals, U = sp_eigh(R_B, R_D)
    except Exception:
        # we will use eig(inv(R_D) R_B)
        w, v = np.linalg.eig(np.linalg.pinv(R_D) @ R_B)
        idx = np.argsort(-np.real(w))
        lambda_vals = np.real(w[idx])
        U = v[:, idx]

    idx_sort = np.argsort(-np.real(lambda_vals))
    lambda_vals = np.real(lambda_vals[idx_sort])
    U = U[:, idx_sort]
    V_use = min(V, len(lambda_vals))

    # r_B via torch
    d_B = np.ones((N, M_b), dtype=np.float32) * target_amplitude
    r_B = compute_rB_torch(bright_zone_mics_index, x, IR, d_B, N, J, n_srcs, device=device)

    # compute q
    q_vec = np.zeros_like(r_B)
    for v in range(V_use):
        weight = lambda_vals[v] / (lambda_vals[v] + mu)
        proj = float(np.dot(U[:, v].T, r_B))
        q_vec += weight * proj * U[:, v]

    q_matrix = q_vec.reshape(n_srcs, J)

    # prepare IR padded output
    IR_prepared = []
    for m in range(len(IR)):
        row = []
        for l in range(len(IR[m])):
            h = np.asarray(IR[m][l], dtype=np.float32)
            row.append(h)
        IR_prepared.append(row)

    print(f"Completed in {time.perf_counter() - t0:.2f} s")
    return q_matrix, IR_prepared

