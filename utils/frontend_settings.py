"""
@file frontend_settings.py
@brief Audio feature generation settings.
"""
from mltk.core.preprocess.audio.audio_feature_generator import AudioFeatureGeneratorSettings

# ==============================================================================
#                             DEFINITIONS
# ==============================================================================

# ---------------------- General settings -------------------
frontend_settings = AudioFeatureGeneratorSettings()
frontend_settings.sample_rate_hz = 16000
frontend_settings.sample_length_ms = 1000

# ---------------------- FFT settings -----------------------
frontend_settings.window_size_ms = 30
frontend_settings.window_step_ms = 10
frontend_settings.filterbank_n_channels = 96
frontend_settings.filterbank_upper_band_limit = 8000
frontend_settings.filterbank_lower_band_limit = 125.0               # The dev board mic seems to have a lot of noise at lower frequencies

# ---------------------- Noise settings ---------------------
frontend_settings.noise_reduction_enable = True                     # Enable the noise reduction block to help ignore background noise in the field
frontend_settings.noise_reduction_smoothing_bits = 10
frontend_settings.noise_reduction_even_smoothing =  0.025
frontend_settings.noise_reduction_odd_smoothing = 0.06
frontend_settings.noise_reduction_min_signal_remaining = 0.10       # Large value to minimize background noise reduction while preserving signal

# ---------------------- DC filter settings ----------------------
frontend_settings.dc_notch_filter_enable = True                     # Enable the DC notch filter to remove DC signal from dev board's mic
frontend_settings.dc_notch_filter_coefficient = 0.95

# ---------------------- Quantization settings -------------------
frontend_settings.quantize_dynamic_scale_enable = True              # Enable dynamic quantization to convert uint16 spectrogram to int8
frontend_settings.quantize_dynamic_scale_range_db = 100.0            # Originally set to 40 dB, we increased it to improve precision