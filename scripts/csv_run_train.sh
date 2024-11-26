#!/bin/bash

python3 src/main.py \
        --model_type vis \
        --ref_file examples/matching_rows_sequoia.csv \
        --save_dir output \
        --cohort TCGA \
        --exp_name run_train_split_0 \
        --batch_size 16 \
        --train \
        --log histo_to_cnv \
        --save_on loss+corr \
        --stop_on loss+corr \
        --split_column split_0

