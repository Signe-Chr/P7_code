import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.io.wavfile as wavfile
from scipy.signal import convolve
#from MLP_regression import FilterNet
from MLP_classification import SoftFilterNet
from scipy.signal import stft
from VAST_filter_coefficients import setup_acoustic_scenario
from Dataset_generator_script import sources_mics, fs_target, rooms #, #mic_directions, mic_positions_list, bright_zone_mics_index
from VAST_filter_coefficients import design_vast_filter


# -------------------------------------------------------------------------
# 1. Define scenario parameters
# -------------------------------------------------------------------------
# Choose segment of audio to evaluate
start = 1
stop = 6

#Input features for prediction
rt60 = 0.27                      # Reverberation, float: np.linspace(0.27, 0.7, 10)
phone_tilt = 1                   # Phone tilt, degrees in radians: 0.261, 0.785, 1.309
user_rotation = 1.57             # Orientation,  degrees in radians: 0, 1.57, 3.14, 4.71
spatial_position = np.array([5, 5, 1.7]).ravel()  # Spatial position (x, y, z): (5, 5 ,1.7) betyder i midten af rummet og i højde 1.7m               # flad ud til 1D

sources_position_list, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index, mic_directions = sources_mics(R= 1 , Center = spatial_position , N_mics=12)

# -------------------------------------------------------------------------
# 2. Generate necessary files
# -------------------------------------------------------------------------


def generate_cut_input(start, stop):

    fs_orig, wav = wavfile.read("relaxing-guitar-loop-v5-245859.wav")
    wav = np.mean(wav, axis=1)
    wav = wav[int(start * fs_orig):int(stop * fs_orig)].astype(np.float32)
    wav /= np.abs(wav).max()
    
    original_path = "Performance Evaluation/input_sound_cut.wav"
    wavfile.write(original_path, fs_orig, (wav * 32767).astype(np.int16))
    print(f'Saved: {original_path}')
    
    return original_path

def generate_IR(): 
    '''
    Generate impulse responses for the given scenario and save to "test_ir.pt" to save time in future runs.
    '''
    room_dim = rooms[1]  # Vælg et rum fra listen
    IR = setup_acoustic_scenario(sources=sources_position_list, mic_positions_list=mic_positions_list, bright_zone_mics_index=bright_zone_mics_index, 
                            dark_zone_mics_index=dark_zone_mics_index, fs_target=fs_target, room_dim=room_dim, 
                            rt60=rt60, mic_directions=mic_directions, user_rotation=user_rotation)[0]
    torch.save(IR, "Performance Evaluation/test_ir.pt")
    return IR

def generate_measured_path():
    '''
    Convolves original input with impulse responses and filter coefficients to get measured output.
    Saves the file "Performance Evaluation/reproduced_sound.wav" to save time in future runs.
    '''

    X = np.concatenate([[rt60], [phone_tilt], [user_rotation], spatial_position])
    dumm = torch.tensor(X, dtype=torch.float32)

    
    input_size = 6 
    L, J = 3, 1024
    output_size = L * J

    #model = FilterNet(input_size, output_size)
    #model.load_state_dict(torch.load("filter_mlp_weights.pth"))

    # Load any needed tensors or metadata
    filters_tensor = torch.load("filters_tensor.pt")  # if you saved it
    input_size = 6  # whatever your input size was
    num_filters = filters_tensor.shape[0]
    filter_dim = filters_tensor.shape[1]

    # Create model
    model = SoftFilterNet(input_size, num_filters, filter_dim, filters_tensor)
    model.load_state_dict(torch.load("mlp_weights.pth", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        Y = model(dumm).cpu().numpy().squeeze()
    q_matrix = Y.reshape(L, J)

    fs_orig, wav = wavfile.read("Performance Evaluation/input_sound_cut.wav")
    IR = torch.load("Performance Evaluation/test_ir.pt", weights_only=False)
    outputs = []
    for i in range(L):
        RIR = IR[bright_zone_mics_index[0]][i]  # Select RIR for bright zone mic and first source
        y = convolve(wav, RIR, mode='full')
        y2 = convolve(y, q_matrix[i], mode='full')
        outputs.append(y2)


    # Align to same length
    min_len = min(len(y) for y in outputs)
    outputs = np.stack([y[:min_len] for y in outputs], axis=1)

    # Normalize to avoid clipping
    outputs /= np.max(np.abs(outputs))

    measured_path = "Performance Evaluation/reproduced_sound.wav"
    wavfile.write(measured_path, fs_orig, (outputs * 32767).astype(np.int16))
    print(f"Saved: {measured_path}")
    return measured_path

def update_all():
    generate_cut_input(start, stop)
    generate_IR()
    generate_measured_path()



# -------------------------------------------------------------------------
# 3. Performance Evaluation
# -------------------------------------------------------------------------
''' Other metrics:
    MSE or RMSE
    Correlation, CC
    RSRQ (SNR lignende)
    SNR, PSNR eller PSIR (peak-snr eller sir)
    Log-Spectral Distance (LSD)
    Cosine similarity
'''

from torch_pesq import PesqLoss

def calculate_pesq(reference: torch.Tensor, degraded: torch.Tensor, sample_rate: int = 44100, mode: float = 0.5):
    """
    Calculate the PESQ MOS score and loss between reference and degraded audio signals.

    Parameters
    ----------
    reference : torch.Tensor
        The reference clean audio signal (shape: [batch, samples]).
    degraded : torch.Tensor
        The degraded/noisy audio signal (shape: [batch, samples]).
    sample_rate : int, optional
        Sampling rate of the audio signals (default: 44100 Hz).
    mode : float, optional
        Weight parameter for PESQLoss (typically between 0 and 1, default: 0.5).

    Returns
    -------
    tuple
        (mos, loss) — both as torch.Tensors.
    """
    pesq = PesqLoss(mode, sample_rate=sample_rate)
    mos = pesq.mos(reference, degraded)
    loss = pesq(reference, degraded)
    return mos, loss

def compute_psnr(original, measured):
    mse = np.mean((original - measured)**2)
    if mse == 0:
        return np.inf
    max_val = np.max(np.abs(original))
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr

def compute_lsd(original, measured, fs, n_fft=1024, hop_length=512):
    # Short-Time Fourier Transform
    f1, t1, Z_orig = stft(original, fs=fs, nperseg=n_fft, noverlap=n_fft-hop_length)
    f2, t2, Z_meas = stft(measured, fs=fs, nperseg=n_fft, noverlap=n_fft-hop_length)
    
    # Power spectrum in dB
    S_orig = 20 * np.log10(np.abs(Z_orig) + 1e-10)
    S_meas = 20 * np.log10(np.abs(Z_meas) + 1e-10)
    
    # LSD per frame
    lsd_frames = np.sqrt(np.mean((S_orig - S_meas)**2, axis=0))
    return np.mean(lsd_frames), lsd_frames  # gennemsnit og alle frames

def compute_CC(original, measured):
    original_mean = np.mean(original)
    measured_mean = np.mean(measured)
    numerator = np.sum((original - original_mean) * (measured - measured_mean))
    denominator = np.sqrt(np.sum((original - original_mean)**2) * np.sum((measured - measured_mean)**2))
    cc = numerator / denominator
    return cc

def compute_cosine_similarity(original, measured):
    dot_product = np.dot(original, measured)
    norm_orig = np.linalg.norm(original)
    norm_meas = np.linalg.norm(measured)
    cosine_similarity = dot_product / (norm_orig * norm_meas)
    return cosine_similarity

def analyze_audio(measured_path, original_path):
    fs_orig, x = wavfile.read(original_path)
    fs_meas, y = wavfile.read(measured_path)

    # Konverter til float mellem -1 og 1, hvis nødvendigt
    if x.dtype != np.float32:
        x = x.astype(np.float32) / np.max(np.abs(x))
    if y.dtype != np.float32:
        y = y.astype(np.float32) / np.max(np.abs(y))
    
    # Convert to mono if multi-channel
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)  # Convert to mono by averaging channels
    if len(x.shape) > 1:
        x = np.mean(x, axis=1)  # Convert to mono if needed

    # Sørg for samme længde
    min_len = min(len(x), len(y))
    x, y = x[:min_len], y[:min_len]

    psnr = compute_psnr(x, y)
    lsd_mean, lsd_frames = compute_lsd(x, y, fs_orig)
    mos_score, pesq_score = calculate_pesq(torch.asarray(x),torch.asarray(y))
    CC_score = compute_CC(x, y)
    Cosine_sim = compute_cosine_similarity(x, y)


    print(f"PSNR: {psnr:.2f} dB")
    print(f"Log-Spectral Distance (LSD): {lsd_mean:.2f} dB")
    print(f"PESQ: {pesq_score}")
    print(f"MOS: {mos_score}")
    print(f"Cross-Correlation (CC): {CC_score:.2f}")
    print(f"Cosine Similarity: {Cosine_sim:.2f}")

    # Plot fejl over tid
    plt.figure(figsize=(10,4))
    plt.plot(lsd_frames)
    plt.title("Log-Spectral Distance per frame")
    plt.xlabel("Frame index")
    plt.ylabel("LSD (dB)")
    plt.grid(True)
    plt.show()

    return psnr, lsd_mean

import numpy as np
from scipy.io import wavfile
from scipy.signal import convolve

def apply_filter_to_audio(filter_path, input_wav_path, output_wav_path=None, normalize=True):
    """
    Applies a FIR filter (from .txt file) to an input WAV file and saves the output.

    Parameters
    ----------
    filter_path : str
        Path to the .txt file containing the filter coefficients.
    input_wav_path : str
        Path to the input WAV file.
    output_wav_path : str, optional
        Path to save the filtered audio. If None, appends '_filtered.wav' to input filename.
    normalize : bool, default=True
        Whether to normalize output to prevent clipping.

    Returns
    -------
    output_wav_path : str
        The path of the saved filtered WAV file.
    """

    # ---- 1. Load filter coefficients ----
    print(np.shape(np.loadtxt(filter_path, dtype=np.float32)))
    filt = np.loadtxt(filter_path, dtype=np.float32).reshape((3, 1024))[0]
    print(f"Loaded filter from '{filter_path}' with {len(filt)} coefficients.")

    # ---- 2. Load input audio ----
    fs, audio = wavfile.read(input_wav_path)
    print(f"Loaded '{input_wav_path}' with sample rate {fs} Hz.")

    # Convert stereo → mono if needed
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Convert to float32 in [-1, 1]
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
        audio /= np.max(np.abs(audio))

    # ---- 3. Apply convolution ----
    #print(np.shape(audio),np.shape(filt))
    filtered_audio = convolve(audio, filt, mode='full')

    # ---- 4. Normalize ----
    if normalize:
        filtered_audio /= np.max(np.abs(filtered_audio))

    # ---- 5. Save output ----
    if output_wav_path is None:
        base = input_wav_path.rsplit('.', 1)[0]
        output_wav_path = f"{base}_filtered.wav"

    wavfile.write(output_wav_path, fs, (filtered_audio * 32767).astype(np.int16))
    print(f"Saved filtered audio to '{output_wav_path}'")

    return output_wav_path


#update_all()
#generate_measured_path()

import Dataset_generator_script as dgs

sources_position_list, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index, mic_directions = dgs.sources_mics(dgs.dark_mic_radius, spatial_position, dgs.N_mics)


q, IR = design_vast_filter(sources_position_list, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index,
                    dgs.wav, dgs.RT60s[0], mic_directions, dgs.user_rotations[0], dgs.fs_target, dgs.J, dgs.N, 
                    dgs.V, dgs.mu, [10, 10, 10], dgs.reg_eps, dgs.target_amplitude)

print(np.shape(q))

np.savetxt("predicted_filter_vast.txt", q, fmt="%.8f")
print(f"Filter saved to 'predicted_filter_vast.txt")

if __name__== "__main__":


    filter1 = "predicted_filter_top1.txt"
    filter2 = "predicted_filter_fnet_2.txt"
    filter3 = "predicted_filter_vast.txt"

    original = "relaxing-guitar-loop-v5-245859.wav"

    output_wav_path = "relaxing-guitar-loop-v5-245859_filterd.wav"

    measured = apply_filter_to_audio(filter1, original, output_wav_path)

    analyze_audio(original, measured)

