#!/bin/bash
#SBATCH -J Sequoia1
#!/bin/bash
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --mail-user=rehan.zuberi@cruk.cam.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p cuda
#SBATCH --gres=gpu:0
#SBATCH -o slurm/out/%J.out
#SBATCH -e slurm/error/%J.error

export PYTHONPATH="/scratchc/fmlab/zuberi01/phd/sequoia-pub:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0 python3 /scratchc/fmlab/zuberi01/phd/sequoia-pub/src/main.py \
    --model_type vis \
    --ref_file /scratchc/fmlab/zuberi01/phd/sequoia-pub/examples/matching_rows_sequoia.csv \
    --save_dir output \
    --cohort TCGA \
    --exp_name run_train_split_1 \
    --batch_size 16 \
    --train \
    --log histo_to_cnv \
    --save_on loss+corr \
    --stop_on loss+corr \
    --split_column split_1 \
    --filter_no_features 0 \
    --feature_path /scratchc/fmlab/zuberi01/masters/saved_patches/40x_400/features2 \
    --rna_prefix cnv_ 
