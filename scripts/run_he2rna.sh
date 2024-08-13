#!/bin/bash
python3 src/he2rna.py \
        --path_csv examples/ref_file.csv \
        --subfolder he2rna \
        --exp_name BRCA \
        --lr 1e-3 \
        --checkpoint pretrained_models/model.pt \
        --change_num_genes \
        --num_genes 19198 \
        --log 0
        
