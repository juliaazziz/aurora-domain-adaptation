#!/usr/bin/env python3
"""
@file run.py
@brief Runs a single experiment.

Usage:
    python3 run.py <base_data_dir> <experiment_id>
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        # Silence TensorFlow

from experiments import ExperimentRunner
from parser import get_parser
from utils.printer import *
from utils.tfrecord_handler import *
from utils.frontend_settings import frontend_settings

DEFAULT_RECORD_FILE = "dataset.tfrecord"
DEFAULT_METADATA_FILE = "metadata.csv"

SEED = 42
BATCH_SIZE = 16
VAL_SIZE = 0.7
TRAIN_SIZE = 0.3
TRAIN_SPLITS = [0.4, 0.6]

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    record_dir = f"{args.base_data_dir}/{DEFAULT_RECORD_FILE}"
    metadata_csv = os.path.join(args.base_data_dir, DEFAULT_METADATA_FILE)
    handler = TFRecordHandler(audio_dir=args.base_data_dir,
                              metadata_dir=metadata_csv,
                              fe_settings=frontend_settings)
    if not os.path.exists(record_dir): handler.write_tfrecord(record_dir)

    print_msg(LogLevel.INFO, f"Running experiment with id {args.experiment_id}")

    dataset = handler.create_batched_dataset(record_dir, 1)
    ds_len = get_dataset_length(dataset)
    ds_shuffled = dataset.shuffle(ds_len, seed=SEED + args.experiment_id)

    train_len = int(TRAIN_SIZE * ds_len)
    train_ds = ds_shuffled.take(train_len)
    val_ds = ds_shuffled.skip(train_len).unbatch().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print_msg(LogLevel.INFO, f"Train dataset length: {get_dataset_length(train_ds)}. Validation dataset length: {get_dataset_length(val_ds)}")

    runner = ExperimentRunner(dataset=dataset)
    runner.run(train_ds=train_ds, val_ds=val_ds, train_splits=TRAIN_SPLITS, split_id=args.experiment_id)

    print_msg(LogLevel.OK, "Experiment completed successfully")

if __name__ == "__main__":
    main()
