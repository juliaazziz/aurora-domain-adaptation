"""
@file tfrecord_handler.py
@brief Class for creating and handling TF Records.
"""
import sys
import os
sys.path.append(".")

import tensorflow as tf
import pandas as pd
from .audio import *
from .log_mel_spectrogram import *
from random import shuffle

class TFRecordHandler:
    def __init__(self, audio_dir, metadata_dir, fe_settings, sr=16000, window_params=None, spec_params=None):
        """
        Initialize the TFRecordHandler.

        Args:
            `audio_dir` (str): Path to the directory containing audio files.
            `metadata_dir` (str): Path to the metadata CSV file.
            `fe_settings` (dict): Frontend settings for audio feature generation.
            `sr` (int, optional): Sampling rate (default 16 kHz).
            `window_params` (dict, optional): Parameters for audio windowing.
            `spec_params` (dict, optional): Log-mel spectrogram parameters.
        """
        self.metadata_dir = metadata_dir
        self.fe_settings = fe_settings
        self.sr = sr
        self.window_params = window_params
        self.spec_params = spec_params

        self.set_audio_directory(audio_dir)

    def set_audio_directory(self, audio_dir):
        self.audio_dir = audio_dir
        self.filenames, self.labels = self._get_filenames()

    def _get_filenames(self):
        """
        Get file paths and ground truth labels for each audio file in the directory.

        Returns:
            `filenames` (list): Absolute paths to each file.
            `labels` (list): Ground truth labels.
        """
        metadata = pd.read_csv(self.metadata_dir)
        files = os.listdir(self.audio_dir)

        filtered_df = metadata[metadata['fname'].isin(files)]
        labels = filtered_df["hasbird"].values
        files_df = filtered_df["fname"].values
        filenames = [os.path.join(self.audio_dir, files_df[i]) for i in range(len(files_df))]

        return filenames, labels

    def _create_example(self, spec, target):
        """
        Create a record element holding a log-mel spectrogram, label, and spectrogram shape.

        Args:
            `spec`: Log-mel spectrogram.
            `target`: Target label.

        Returns:
            `example`: TFRecord example.
        """
        feature = {
            'spec': tf.train.Feature(int64_list=tf.train.Int64List(value=spec.flatten())),
            'target': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(target)])),
            'spec_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=spec.shape)),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))

    def write_tfrecord(self, tfrecord_filename):
        """
        Write a TFRecord file containing log-mel spectrograms and labels.

        Args:
            `tfrecord_filename` (str): Path to save the TFRecord file.
        """
        zipped = list(zip(self.filenames, self.labels))
        shuffle(zipped)
        filenames, labels = zip(*zipped)

        with tf.io.TFRecordWriter(tfrecord_filename) as writer:
            for file, label in zip(filenames, labels):
                scale = audio_load_wav_mono(file, return_numpy=True, fs=self.sr)
                # GAIN_DB = 40
                # gain_factor = 10 ** (GAIN_DB / 20)
                # scale *= gain_factor
                windows, _ = audio_extract_windows(scale, self.sr, self.window_params)

                # Process each window
                for window in windows:
                    spec = log_mel_spectrogram_mltk(window, self.fe_settings)
                    example = self._create_example(spec, label)
                    writer.write(example.SerializeToString())

        print(f'TFRecord file {tfrecord_filename} created successfully.')

    @staticmethod
    def parse_tfrecord(example):
        """
        Parse a record containing (spectrogram, target).

        Args:
            `example`: TFRecord element.

        Returns:
            `spec`: Log-mel spectrogram.
            `target`: Label.
        """
        feature_description = {
            'spec': tf.io.VarLenFeature(tf.int64),
            'target': tf.io.FixedLenFeature([], tf.int64),
            'spec_shape': tf.io.FixedLenFeature([2], tf.int64)
        }

        example = tf.io.parse_single_example(example, feature_description)
        spec_shape = example['spec_shape']
        spec = tf.sparse.to_dense(example['spec'])
        spec = tf.reshape(spec, spec_shape)

        target = tf.cast(example['target'], tf.int64)
        target = tf.squeeze(target)

        return spec, target

    @staticmethod
    def create_batched_dataset(tfrecord_filename, batch_size=32):
        """
        Create a batched dataset from a TFRecord file.

        Args:
            `tfrecord_filename` (str): Path to the TFRecord file.
            `batch_size` (int, optional): Batch size (default: 32).

        Returns:
            `dataset`: Batched TensorFlow dataset.
        """
        raw_dataset = tf.data.TFRecordDataset(tfrecord_filename)
        dataset = raw_dataset.map(TFRecordHandler.parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

        if batch_size is not None:
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    @staticmethod
    def extract_labels(dataset):
        """
        Extract labels from a TensorFlow dataset.

        Args:
            `dataset`: Dataset with elements in the form (feature, label).

        Returns:
            `targets` (np.array): Array of labels.
        """
        targets = []

        for _, target in dataset:
            targets.append(target)

        targets = tf.concat(targets, axis=0)

        return targets.numpy()

def get_dataset_length(dataset):
    return dataset.reduce(tf.constant(0, dtype=tf.int32),
                            lambda x, batch: x + tf.shape(batch[0])[0]).numpy()