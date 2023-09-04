#!/usr/bin/bash

python3 pre_processing/patch_gen_hdf5.py \
        --ref_file ./examples/ref_file.csv \
        --wsi_path ./examples/HE \
        --patch_path ./examples/Patches_hdf5 \
        --mask_path ./examples/Patches_hdf5 \
        --patch_size 256 \
        --max_patches_per_slide 4000