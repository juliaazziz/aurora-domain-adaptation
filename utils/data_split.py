"""
@file data_split.py
@brief Functions to split training, validation and test data.
"""
import os
import random
import wave
from collections import defaultdict

# ==================================================================================
#                                   PRIVATE FUNCTIONS
# ==================================================================================

def _get_wav_duration(file_path):
    """
    Calculate the duration of a WAV file in seconds.
    
    Args:
        `file_path` (str): path to the WAV file.
    Returns:
        `duration` (float): duration of the WAV file in seconds.
    """
    with wave.open(file_path, 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        return frames / float(rate)

# ==================================================================================
#                                   PUBLIC FUNCTIONS
# ==================================================================================

def data_split_by_duration(filenames, folder, labels, test_size=0.2, stratify=False, random_state=42):
    """
    Splits filenames and labels into train and test sets based on total WAV duration.

    Args:
        `filenames` (list): list of WAV file names to split.
        `folder` (str): path to the folder containing the WAV files.
        `labels` (list): list of corresponding labels for each file.
        `test_size` (float): proportion of the total duration to allocate for the test set (default: 0.2).
        `stratify` (bool): if specified, ensures the split is stratified by label (default: False).
        `random_state` (int): seed for random operations (default: 42).
    Returns:
        A tuple containing:
        `train_files`: list of file names in the training set.
        `test_files`: list of file names in the test set.
        `train_labels`: list of labels corresponding to the training files.
        `test_labels`: list of labels corresponding to the test files.
    """
    random.seed(random_state)
    
    if stratify:
        grouped_files = defaultdict(list)
        for fname, label in zip(filenames, labels):
            grouped_files[label].append(fname)
    else:
        grouped_files = {None: filenames}
    
    train_files, train_labels = [], []
    test_files, test_labels = [], []
    
    for label, group_files in grouped_files.items():
        random.shuffle(group_files)
        
        total_duration = sum(_get_wav_duration(os.path.join(folder, f)) for f in group_files)
        target_test_duration = total_duration * test_size
        current_test_duration = 0
        
        for fname in group_files:
            file_duration = _get_wav_duration(os.path.join(folder, fname))
            if current_test_duration + file_duration <= target_test_duration:
                test_files.append(fname)
                test_labels.append(label)
                current_test_duration += file_duration
            else:
                train_files.append(fname)
                train_labels.append(label)
    
    return train_files, test_files, train_labels, test_labels
