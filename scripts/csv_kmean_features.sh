#!/bin/bash
python3 /scratchc/fmlab/zuberi01/phd/sequoia-pub/pre_processing/kmean_features.py \
        --ref_file /scratchc/fmlab/zuberi01/phd/sequoia-pub/examples/matching_rows_sequoia.csv  \
        --patch_data_path /scratchc/fmlab/zuberi01/masters/saved_patches/40x_400/features2/h5_files \
        --feature_path /scratchc/fmlab/zuberi01/masters/saved_patches/40x_400/features2  \
        --num_clusters 100 \
        --feature_name 'features'