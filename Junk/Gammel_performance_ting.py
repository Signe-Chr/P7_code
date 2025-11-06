from VAST_filter_coefficients import setup_acoustic_scenario
from Dataset_generator_script import sources_mics, fs_target, rooms #, #mic_directions, mic_positions_list, bright_zone_mics_index



"""
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


def generate_cut_input(start, stop, input):

    fs_orig, wav = wavfile.read(input)
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

def generate_measured_path(original):
    '''
    Convolves original input with impulse responses and filter coefficients to get measured output.
    Saves the file "Performance Evaluation/reproduced_sound.wav" to save time in future runs.
    '''
    cut = original
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
"""



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
    mos_score, pesq_score = compute_pesq(torch.asarray(x),torch.asarray(y))
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




"""
if __name__== "__main__":


    filter1 = "predicted_filter_top1.txt"
    filter2 = "predicted_filter_fnet_2.txt"
    filter3 = "predicted_filter_vast.txt"

    original = "relaxing-guitar-loop-v5-245859.wav"

    output_wav_path = "relaxing-guitar-loop-v5-245859_filterd.wav"
    #generate_cut_input(start, stop, original)

    #measured = apply_filter_to_audio(filter1, original, output_wav_path)

    #analyze_audio(original, measured)

    file = "Phone Zone Data/Proc_B4107_M0_P0R000_T0.mat"
    data = loadmat(file)
    print(data.keys())
"""




def performance_evaluation(
    test_features, test_filters, test_RIRs,
    reference, fs_wav,
    bright_zone_mics_index, dark_zone_mics_index, save=False
):
    """
    Simulates pressure fields for all test samples, saves degraded audio
    for bright and dark zones, and computes perceptual metrics.
    """
    save_dir="Performance Evaluation"
    os.makedirs(save_dir, exist_ok=True)

    # Save reference (reference) audio for comparison
    ref_path = os.path.join(save_dir, "reference.wav")
    ref_np = reference.squeeze().cpu().numpy()
    ref_np /= np.max(np.abs(ref_np))
    ref_np = np.asarray(ref_np, dtype=np.float32).ravel()  # <-- gør 1D
    #reference = ref_np[:]

    wavfile.write(ref_path, fs_wav, (ref_np * 32767).astype(np.int16))



    results = []

    for i in range(len(test_features)):
        print(f"\n--- Evaluating sample {i+1}/{len(test_features)} ---")

        rir = test_RIRs[i]
        filter = test_filters[i]
        rir = rir.float().to(reference.device)
        filter = filter.float().to(reference.device)
        reference = reference.float().to(reference.device)

        # --- 1. Compute acoustic pressure ---
        p = compute_pressure_with_input(rir, filter, reference)
        
        # --- 2. Extract bright & dark zone pressures ---
        p_bright = p[bright_zone_mics_index[i]]
        p_dark   = p[dark_zone_mics_index[i]]

        #p_bright_mean = torch.mean(p_bright, dim=0)
        p_dark_mean = torch.mean(p_dark, dim=0)

        # --- 3. Convert to same type/shape as reference ---
        p_bright_t = p_bright.to(dtype=reference.dtype, device=reference.device)
        p_dark_t   = p_dark_mean.unsqueeze(0).to(dtype=reference.dtype, device=reference.device)

        # Normalize (match input scaling)
        p_bright_t = p_bright_t / torch.max(torch.abs(p_bright_t))
        p_dark_t   = p_dark_t / torch.max(torch.abs(p_dark_t))

        
        # --- 4. Save degraded WAVs ---
        if save == True:
            bright_path = os.path.join(save_dir, f"degraded_bright_{i}.wav")
            dark_path   = os.path.join(save_dir, f"degraded_dark_{i}.wav")

        # Convert to NumPy and ensure 2D for WAV: [N_samples, N_channels]
        p_bright_np = p_bright_t.cpu().numpy().reshape(-1, 1)
        p_dark_np   = p_dark_t.cpu().numpy().reshape(-1, 1)
        p_bright_np = np.asarray(p_bright_np, dtype=np.float32).ravel() # <-- gør 1D
        p_dark_np   = np.asarray(p_dark_np, dtype=np.float32).ravel() # <-- gør 1D

        wavfile.write(bright_path, fs_wav, (p_bright_np * 32767).astype(np.int16))
        wavfile.write(dark_path,   fs_wav, (p_dark_np   * 32767).astype(np.int16))

        # --- 5. Compute metrics ---
        pesq_b = compute_pesq(ref_path, bright_path)
        pesq_d = compute_pesq(ref_path, dark_path)

        stoi_b = compute_STOI(ref_path, bright_path)
        stoi_d = compute_STOI(ref_path, dark_path)

        psnr_b = compute_psnr(ref_np, p_bright_np)
        psnr_d = compute_psnr(ref_np, p_dark_np)

        cc_b = compute_CC(ref_np, p_bright_np)
        cc_d = compute_CC(ref_np, p_dark_np)

        ac = acoustic_contrast(rir, filter, reference, bright_zone_mics_index, dark_zone_mics_index)

        # --- 6. Store results ---
        results.append({
            "sample_idx": i,
            "PESQ_bright": pesq_b,
            "PESQ_dark": pesq_d,
            "STOI_bright": stoi_b,
            "STOI_dark": stoi_d,
            "PSNR_bright": psnr_b,
            "PSNR_dark": psnr_d,
            "CC_bright": cc_b,
            "CC_dark": cc_d,
            "Acoustic_Contrast": ac.item() if torch.is_tensor(ac) else ac
        })

        print(f"Results: PESQ_b={pesq_b:.2f}, PESQ_d={pesq_d:.2f}, STOI_b={stoi_b:.2f}, STOI_d={stoi_d:.2f}")
        print(f"         CC_b={cc_b:.2f},     CC_d={cc_d:.2f},     PSNR_b={psnr_b:.2f}, PSNR_d={psnr_d:.2f}")
        print(f"         AC={ac:.2f}")

    return results







#########################################################3

def compute_pressure_with_input2(rir: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    Simulates the acoustic pressure at all mics by convolving RIRs directly with the input signal.

    Parameters:
        rir: [n_mics, n_srcs, n_rir_samples]
        reference: [1, n_input_samples] (The source signal)
    
    Returns:
        p: [n_mics, n_output_samples] (Acoustic pressure)
    """
    n_mics, n_srcs, n_rir_samples = rir.shape
    n_input_samples = reference.shape[-1]
    
    # Output length = n_rir_samples + n_input_samples - 1
    output_len = n_rir_samples + n_input_samples - 1
    
    # Zero pad input
    reference_padded = F.pad(reference, (0, output_len - n_input_samples), 'constant', 0)
    p = torch.zeros((n_mics, output_len), device=rir.device)

    # FFT length (power of 2 for efficiency)
    n_fft = 2 ** int(np.ceil(np.log2(output_len)))
    X_fft = torch.fft.rfft(reference_padded, n=n_fft).squeeze(0)

    # Loop through microphones and sources
    for m in range(n_mics):
        p_m = torch.zeros(output_len, device=rir.device)
        for s in range(n_srcs):
            h = rir[m, s, :]  # [n_rir_samples]
            h_padded = F.pad(h, (0, output_len - n_rir_samples), 'constant', 0)
            H_fft = torch.fft.rfft(h_padded, n=n_fft)
            
            # Convolution via multiplication in frequency domain
            P_fft = H_fft * X_fft
            p_m_s = torch.fft.irfft(P_fft, n=n_fft)[:output_len]
            p_m += p_m_s

        p[m, :] = p_m

    return p

def generate_measured_path(filters, unfiltered):
    '''
    Convolves the received sound with the filters.
    '''
    print(filters)
    print(unfiltered)
    filtered = []
    for i in range(len(filters)):  # For each source
        y = convolve(filters[i], unfiltered[i], mode='full')
        filtered.append(y)

    # Align to same length
    min_len = min(len(y) for y in filtered)
    filtered = np.stack([y[:min_len] for y in filtered], axis=1)

    # Normalize to avoid clipping
    filtered /= np.max(np.abs(filtered))
    return filtered

def performance_evaluation2(
    test_features, filters, RIRs,
    reference, fs_wav,
    bright_zone_mics_index, dark_zone_mics_index, save=False
):
    """
    Simulates pressure fields for all test samples, saves degraded audio
    for bright and dark zones, and computes perceptual metrics.
    """
    save_dir="Performance Evaluation"
    os.makedirs(save_dir, exist_ok=True)

    # Save reference (reference) audio for comparison
    ref_path = os.path.join(save_dir, "reference.wav")


    ref_np = reference.squeeze().cpu().numpy()
    ref_np /= np.max(np.abs(ref_np))
    ref_np = np.asarray(ref_np, dtype=np.float32).ravel()  # <-- gør 1D
    #reference = ref_np[:]

    wavfile.write(ref_path, fs_wav, (ref_np * 32767).astype(np.int16))
    
    unfiltered = compute_pressure_with_input2(RIRs[0], reference).cpu().numpy()

    results = []

    for i in range(len(test_features)):
        print(f"\n--- Evaluating sample {i+1}/{len(test_features)} ---")

        rir = RIRs[i]
        filter = filters[i]
        rir = rir.float().to(reference.device)
        filter = filter.float().to(reference.device)
        reference = reference.float().to(reference.device)

        # --- 1. Compute acoustic pressure ---
        filtered = generate_measured_path(filter, unfiltered)
        
        # --- 2. Extract bright & dark zone pressures ---
        p_bright = filtered[bright_zone_mics_index[i]]
        p_dark   = filtered[dark_zone_mics_index[i]]
        p_dark_mean = torch.mean(p_dark, dim=0)

        # --- 3. Convert to same type/shape as reference ---
        p_bright_t = p_bright.to(dtype=reference.dtype, device=reference.device)
        p_dark_t   = p_dark_mean.unsqueeze(0).to(dtype=reference.dtype, device=reference.device)

        # Normalize (match input scaling)
        p_bright_t = p_bright_t / torch.max(torch.abs(p_bright_t))
        p_dark_t   = p_dark_t / torch.max(torch.abs(p_dark_t))

        
        # --- 4. Save degraded WAVs ---
        if save == True:
            bright_path = os.path.join(save_dir, f"degraded_bright_{i}.wav")
            dark_path   = os.path.join(save_dir, f"degraded_dark_{i}.wav")

        # Convert to NumPy and ensure 2D for WAV: [N_samples, N_channels]
        p_bright_np = p_bright_t.cpu().numpy().reshape(-1, 1)
        p_dark_np   = p_dark_t.cpu().numpy().reshape(-1, 1)
        p_bright_np = np.asarray(p_bright_np, dtype=np.float32).ravel() # <-- gør 1D
        p_dark_np   = np.asarray(p_dark_np, dtype=np.float32).ravel() # <-- gør 1D

        wavfile.write(bright_path, fs_wav, (p_bright_np * 32767).astype(np.int16))
        wavfile.write(dark_path,   fs_wav, (p_dark_np   * 32767).astype(np.int16))

        # --- 5. Compute metrics ---
        pesq_b = compute_pesq(ref_path, bright_path)
        pesq_d = compute_pesq(ref_path, dark_path)

        stoi_b = compute_STOI(ref_path, bright_path)
        stoi_d = compute_STOI(ref_path, dark_path)

        psnr_b = compute_psnr(ref_np, p_bright_np)
        psnr_d = compute_psnr(ref_np, p_dark_np)

        cc_b = compute_CC(ref_np, p_bright_np)
        cc_d = compute_CC(ref_np, p_dark_np)

        ac = acoustic_contrast(rir, filter, reference, bright_zone_mics_index, dark_zone_mics_index)

        # --- 6. Store results ---
        results.append({
            "sample_idx": i,
            "PESQ_bright": pesq_b,
            "PESQ_dark": pesq_d,
            "STOI_bright": stoi_b,
            "STOI_dark": stoi_d,
            "PSNR_bright": psnr_b,
            "PSNR_dark": psnr_d,
            "CC_bright": cc_b,
            "CC_dark": cc_d,
            "Acoustic_Contrast": ac.item() if torch.is_tensor(ac) else ac
        })

        print(f"Results: PESQ_b={pesq_b:.2f}, PESQ_d={pesq_d:.2f}, STOI_b={stoi_b:.2f}, STOI_d={stoi_d:.2f}")
        print(f"         CC_b={cc_b:.2f},     CC_d={cc_d:.2f},     PSNR_b={psnr_b:.2f}, PSNR_d={psnr_d:.2f}")
        print(f"         AC={ac:.2f}")

    return results





#print(RIRs.shape)
if __name__== "__main__":
    X, filters, bright_zone_mics_index, dark_zone_mics_index, n_srcs, RIRs= load_data()[0]
    device = load_data()[1]
    bright_zone_mics_index = np.array(bright_zone_mics_index).T
    dark_zone_mics_index = np.array(dark_zone_mics_index).T

    X_test=np.stack([X[0],X[1]],axis=0)
    n_srcs = n_srcs[0]
    filter_len = len(filters[0])//n_srcs
    # For the first two test points:
    filter_test = torch.stack([
        filters[0].reshape(n_srcs, filter_len),
        filters[1].reshape(n_srcs, filter_len)
    ], dim=0)
    test_RIRs = torch.stack([RIRs[0], RIRs[1]], dim=0).to(device)  # shape [2, 13, 3, 512]

    wav_path = "relaxing-guitar-loop-v5-245859.wav"
    fs_wav, wav = wavfile.read(wav_path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav = wav[5*fs_wav : 7*fs_wav]
    wav = wav / np.max(np.abs(wav))  # scale to [-1,1]
    reference = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    reference = reference.to(device)
    # reference tensor
    bright_tensor = bright_zone_mics_index[0]  # the only element in the list
    dark_tensor   = dark_zone_mics_index[0]

    # Select first two test points (assuming first axis corresponds to data points)
    bright_zone_mics_index_test = [bright_zone_mics_index[0], bright_zone_mics_index[1]]
    dark_zone_mics_index_test   = [dark_zone_mics_index[0], dark_zone_mics_index[1]]

    performance_evaluation2(X_test, filter_test, test_RIRs, reference, fs_wav, bright_zone_mics_index_test, dark_zone_mics_index_test)

