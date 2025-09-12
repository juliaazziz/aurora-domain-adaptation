"""
@file log_mel_spectrogram.py
@brief Functions to compute different variations of log-mel spectrograms.
"""
import sys
sys.path.append(".")

import tensorflow as tf
import numpy as np
from mltk.core.preprocess.utils import audio as audio_utils
from .frontend_settings import frontend_settings

# ==================================================================================
#                                   PUBLIC FUNCTIONS
# ==================================================================================

def log_mel_spectrogram(audio, sr=16000, spec_params=None, return_numpy=True):
    """
    Compute log-mel spectrogram of an audio signal using TensorFlow.

    Args:
        `audio` (array): audio signal.
        `sr` (int): sampling rate.
        `spec_params` (dict): log-mel spectrogram parameters.
        `return_numpy` (bool): return spectrogram as a numpy array.

    Returns:
        `log_mel_spectrograms` (array): log-mel spectrogram.
    """

    if spec_params is None:
        spec_params = {
            "n_mels": 96,
            "frame_length": 30e-3,
            "frame_step": 10e-3,
            "fft_length": 1024,
            "upper_f": 8000
    }

    stfts = tf.signal.stft(audio,
                           frame_length=int(sr*spec_params["frame_length"]),
                           frame_step=int(sr*spec_params["frame_step"]),)

    spectrogram = tf.abs(stfts)

    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz = 125, spec_params["upper_f"]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(spec_params["n_mels"],
                                                                        num_spectrogram_bins,
                                                                        sr,
                                                                        lower_edge_hertz,
                                                                        upper_edge_hertz)

    mel_spectrograms = tf.tensordot(spectrogram,
                                    linear_to_mel_weight_matrix,
                                    1)
    mel_spectrograms.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    
    if return_numpy:
        log_mel_spectrograms = log_mel_spectrograms.numpy()

    return log_mel_spectrograms

def log_mel_spectrogram_uint16(audio, sr=16000, spec_params=None):
    """
    Compute log-mel spectrogram of an audio signal in uint16 using
    TensorFlow.

    Args:
        `audio` (array): audio signal.
        `sr` (int): sampling rate.
        `spec_params` (dict): log-mel spectrogram parameters.

    Returns:
        `log_mel_spectrograms` (array): log-mel spectrogram in uint16.
    """
    log_mel_spectrograms = log_mel_spectrogram(audio, sr, spec_params, return_numpy=False)

    min_val = tf.reduce_min(log_mel_spectrograms)
    max_val = tf.reduce_max(log_mel_spectrograms)
    
    log_mel_spectrograms_normalized = (log_mel_spectrograms - min_val) / (max_val - min_val)
    
    log_mel_spectrograms_uint16 = tf.cast(log_mel_spectrograms_normalized * 65535, tf.uint16)
    
    log_mel_spectrograms = log_mel_spectrograms_uint16.numpy()

    return log_mel_spectrograms

def log_mel_spectrogram_int8(audio, sr=16000, spec_params=None, dynamic_range=800):
    """
    Compute log-mel spectrogram of an audio signal and convert it to int8 format.

    Args:
        `audio` (array): audio signal.
        `sr` (int): sampling rate.
        `spec_params` (dict): log-mel spectrogram parameters.
        `dynamic_range` (int): dynamic range for quantization.

    Returns:
        `log_mel_spectrograms` (array): log-mel spectrogram in int8.
    """
    log_mel_spectrograms = log_mel_spectrogram(audio, sr, spec_params)
    log_mel_spectrograms = _uint16_to_int8(log_mel_spectrograms, dynamic_range)

    return log_mel_spectrograms

def log_mel_spectrogram_mltk(audio, fe_settings=frontend_settings, dtype=np.int8):
    """
    Compute log-mel spectrogram of an audio signal using Silicon Labs' MLTK API.

    Args:
        `audio` (array): audio signal.
        `fe_settings` (dict): frontend settings for audio feature generation.
        `dtype` (dtype): spectrogram data type (default is int8).

    Returns:
        `log_mel_spectrogram` (array): log-mel spectrogram.
    """
    log_mel_spectrogram = audio_utils.apply_frontend(sample=audio,
                                                     settings=fe_settings,
                                                     dtype=dtype)
    return log_mel_spectrogram

# ==================================================================================
#                                   PRIVATE FUNCTIONS
# ==================================================================================

def _uint16_to_int8(spec, dynamic_range=300):
    """
    Convert an array from uint16 to int8 format using dynamic scaling.
    
    Max value is mapped to +127, and the max value - <dynamic_range>
    is mapped to -128. Anything below max value - <dynamic_range> is
    mapped to -128. 

    Note: ported to Python from Silicon Labs audio feature generation C++
    library.

    Args:
        `spec` (array): uint16 array to be converted.
        `dynamic_range` (int): dynamic range to use for conversion
                               (default is 300).

    Returns:
        `spec` (array): converted int8 arrray.
    """
    maxval = np.max(spec)
    minval = min(maxval - dynamic_range, 0)
    val_range = max(maxval - minval, 1)

    spec -= minval
    spec *= 255

    spec /= val_range
    spec -= 128
    spec = np.clip(spec, -128, 127)

    spec = np.int8(spec)

    return spec