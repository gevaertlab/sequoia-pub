#!/bin/bash

python3 pre_processing/compute_resnet_features_hdf5.py \
        --ref_file ./examples/ref_file.csv \
        --patch_data_path ./examples/Patches_hdf5 \
        --feature_path ./examples/features \
        --max_patch_number 4000