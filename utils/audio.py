"""
@file audio.py
@brief Functions to work with WAV audio files.
"""
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from scipy.signal import welch

# ==================================================================================
#                                   PUBLIC FUNCTIONS
# ==================================================================================

def audio_load_wav_mono(filename, return_numpy=False, fs=16000):
    """
    Load and resample a WAV file into single-channel audio.
    Returns either a tensor or a numpy array, depending on
    `return_numpy`.

    Args:
        `filename` (str): WAV file to load.
        `return_numpy` (bool): return audio signal as a numpy array.

    Returns:
        `wav` (tensor or array): prepared audio signal.
    """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=fs)

    # cast to numpy array if necessary
    if return_numpy:
        wav = wav.numpy()

    return wav

def audio_extract_windows(audio, sr=16000, window_params=None):
    """
    Extract fixed-length windows from an audio file.

    Args:
        `audio` (list): audio signal.
        `window_params` (dict): dictionary holding the following parameters:
        `window_size` (float): size of the window in seconds (default 1.0 s).
        `overlap` (float): overlapping ratio between consecutive windows (default 0.5).
        `sr` (int): sampling rate for loading the audio file (default 16 kHz).
    
    Returns:
        `windows` (list): list of audio windows (numpy arrays).
        `n_windows` (int): number of extracted windows.
    """
    window_size, overlap = 1.0, 0
    if window_params is not None:
        window_size, overlap = window_params.values()

    window_samples = int(window_size * sr)
    step_size = int(window_samples * (1 - overlap))
    
    windows = []

    if len(audio) < window_samples:
            last_window = audio[-window_samples:]
            last_window = np.pad(last_window, (0, window_samples - len(last_window)), mode='constant')
            windows.append(last_window)
    else:
        for start in range(0, len(audio) - window_samples + 1, step_size):
            window = audio[start:start + window_samples]
            windows.append(window)

        if (len(audio) % window_samples):
            windows.append(audio[-window_samples:])
    
    return windows, len(windows)

def audio_load_wav_for_map(filename, label):
    """
    Wrapper for audio_load_wav_mono to be used with `map` method in
    a TensorFlow Dataset.

    Args:
        `filename` (str): WAV file to load.
        `label`: true label for the WAV file.

    Returns:
        `wav`: prepared audio file.
        `label`: true label for the WAV file.
    """

    return audio_load_wav_mono(filename), label

def audio_low_frequency_energy_ratio(data, sr=16000, threshold_hz=100):
    """
    Compute ratio of low-frequency to total energy.

    Args:
        `data` (array): audio signal.
        `sr` (int): sampling rate for audio signal (default 16 kHz).
        `threshold_hz` (int): upper threshold for low frequencies (default 100 Hz).

    Returns:
        `low_freq_ratio` (int): ratio of low-frequency to total energy.
    """
    frequencies, psd = welch(data, fs=sr, nperseg=512)

    low_freq_energy = np.sum(psd[frequencies < threshold_hz])
    total_energy = np.sum(psd)

    low_freq_ratio = low_freq_energy / total_energy
    return low_freq_ratio

def audio_zero_crossing_rate(audio_buffer):
    """
    Compute the zero-crossing rate of an audio buffer.
    
    Parameters:
        `audio_buffer` (array): audio signal.
        
    Returns:
        `zcr` (float): zero-crossing rate (per second or per buffer).
    """
    zero_crossings = np.sum(np.diff(np.sign(audio_buffer)) != 0)
    zcr = zero_crossings / len(audio_buffer)
    return zcr

def audio_spectral_flatness(audio_buffer, sr=16000):
    """
    Compute the spectral flatness of an audio signal.
    
    Parameters:
        `audio_buffer` (array): audio signal.
        `sr` (int): sampling rate of the audio signal (default 16000 Hz).
        
    Returns:
        `sf` (float): spectral flatness value (0 to 1).
    """
    # Compute PSD
    freqs, psd = welch(audio_buffer, fs=sr, nperseg=512)
    
    # Avoid division by zero in log calculation
    psd[psd == 0] = np.finfo(float).eps
    
    # Compute geometric and arithmetic mean of the PSD
    geometric_mean = np.exp(np.mean(np.log(psd)))
    arithmetic_mean = np.mean(psd)
    
    # Calculate spectral flatness
    sf = geometric_mean / arithmetic_mean

    return sf