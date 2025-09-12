import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Run an experiment")
    parser.add_argument('base_data_dir', type=str, help='Path to the base data directory')
    parser.add_argument('-experiment_id', type=int, default=0, help='ID of the experiment to run (optional)')

    return parser