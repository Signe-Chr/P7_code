import librosa
import numpy as np
import os

def analyze_sound_pressure(audio_path, target_sr=22050):
    """
    Loads an audio file and calculates the Root Mean Square (RMS) digital 
    amplitude and the Sound Pressure Level (SPL) relative to a full-scale digital signal.

    Args:
        audio_path (str): Path to the audio file (e.g., .wav, .mp3).
        target_sr (int): Sample rate to resample the audio (standard for librosa).

    Returns:
        dict: Contains RMS amplitude and estimated SPL in dB.
    """
    if not os.path.exists(audio_path):
        return {"error": f"File not found: {audio_path}"}
    
    try:
        # Load the audio data. 'y' is the amplitude signal, 'sr' is the sample rate.
        # librosa automatically converts the data to floating point numbers [-1.0, 1.0].
        y, sr = librosa.load(audio_path, sr=target_sr)
    except Exception as e:
        return {"error": f"Could not load audio file: {e}"}

    # --- 1. Calculate RMS Amplitude ---
    # RMS is the square root of the mean of the squares of the values.
    # This is a measure of the effective energy or pressure of the signal.
    rms_amplitude = np.sqrt(np.mean(y**2))

    # --- 2. Calculate Sound Pressure Level (SPL in dB) ---
    
    # We define the reference pressure (P_ref) as the maximum possible digital amplitude (1.0).
    P_ref_digital = 1.0
    
    # SPL (dB) is calculated as: 20 * log10(P_rms / P_ref)
    # Adding a small constant (1e-6) to the RMS to prevent log(0) errors for silent clips.
    if rms_amplitude > 1e-6:
        spl_db = 20.0 * np.log10(rms_amplitude / P_ref_digital)
    else:
        # If the clip is silent, return a very low dB value
        spl_db = -120.0 

    return {
        "rms_amplitude": rms_amplitude,
        "spl_db": spl_db,
        "sample_rate": sr,
        "duration_seconds": librosa.get_duration(y=y, sr=sr)
    }

# --- Example Usage ---
# NOTE: Replace 'your_sound_clip.wav' with the actual path to your audio file.
audio_file_path = 'Signe_sang'

# For demonstration, assume a file path exists:
if not os.path.exists(audio_file_path):
    # Create a dummy signal if the file isn't found, so the example runs
    print(f"File '{audio_file_path}' not found. Analyzing a dummy signal instead.")
    # Create a 440 Hz sine wave signal (A4 note)
    sr = 22050
    duration = 3 # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y_dummy = 0.5 * np.sin(2 * np.pi * 440 * t) 
    # Calculate dummy RMS and SPL manually for printing
    rms_dummy = np.sqrt(np.mean(y_dummy**2))
    spl_dummy = 20.0 * np.log10(rms_dummy / 1.0)
    
    print(f"RMS Amplitude (Digital Energy): {rms_dummy:.4f}")
    print(f"Estimated SPL (Digital Reference): {spl_dummy:.2f} dBFS")
    
else:
    analysis_results = analyze_sound_pressure(audio_file_path)

    if "error" in analysis_results:
        print(analysis_results["error"])
    else:
        print("\n--- Sound Pressure Analysis Results ---")
        print(f"Audio Duration: {analysis_results['duration_seconds']:.2f} seconds")
        print(f"Sample Rate: {analysis_results['sample_rate']} Hz")
        print("-" * 35)
        # RMS Amplitude: Represents the effective digital energy
        print(f"RMS Amplitude (Digital Energy): {analysis_results['rms_amplitude']:.4f}")
        
        # SPL in dBFS (Decibels Full Scale): Relative to max digital volume (1.0)
        # This is a measure of "loudness" within the digital domain.
        print(f"Estimated SPL (Digital Reference): {analysis_results['spl_db']:.2f} dBFS")
        
        # Note on physical pressure:
        print("\nNote: The dBFS value is relative to the maximum possible digital amplitude (0 dBFS).")
        print("To convert to physical pressure (Pascals or dBSPL) requires a calibrated microphone.")
