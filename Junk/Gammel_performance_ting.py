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