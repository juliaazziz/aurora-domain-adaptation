#!/usr/bin/bash
####################################################
# Script to run several DCASE experiments          #
# Usage: ./run.sh [base_data_dir]                  #
####################################################

base_data_dir=$1
if [ -z "$base_data_dir" ]; then
    echo "Usage: $0 [base_data_dir]"
    exit 1
fi

for id in {0..4}
do
    python3 run.py $base_data_dir $id
    if [ $? -ne 0 ]; then
        echo "Experiment $id failed"
        exit 1
    fi
    echo "Experiment $id completed successfully"
done
