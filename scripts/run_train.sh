#!/bin/bash

python3 src/main.py \
        --model_type vis \
        --ref_file examples/ref_file.csv \
        --save_dir output \
        --cohort TCGA \
        --exp_name run_train \
        --batch_size 16 \
        --checkpoint pretrained_models/model_best.pt \
        --k 5 \
        --train \
        --log 0 \
        --change_num_genes \
        --num_genes 19198 \
        --save_on loss+corr \
        --stop_on loss+corr
