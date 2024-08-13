#!/bin/bash

python3 src/visualize.py --study gbm \
                            --project spatial_GBM_pred \
                            --wsi_file_name HRI_251_T.tif \
                            --gene_names $path/gene_ids/top_1000_gbm.npy \
                            --save_folder top_1000_gbm \
                            --model_type vis \
                            --feat_type uni