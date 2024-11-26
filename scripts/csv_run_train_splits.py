import os

# Base SLURM settings without job name
SBATCH_TEMPLATE = """#!/bin/bash
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
"""

# Create output directories
os.makedirs("slurm_scripts", exist_ok=True)
os.makedirs("slurm/out", exist_ok=True)
os.makedirs("slurm/error", exist_ok=True)

# Loop to create and submit jobs
for i in range(10):
    exp_name = f"run_train_split_{i}"
    split_column = f"split_{i}"
    job_name = f"Sequoia{i}"  # Dynamic job name
    
    job_script = f"slurm_scripts/run_job_{i}.sh"
    with open(job_script, "w") as f:
        f.write(f"#!/bin/bash\n#SBATCH -J {job_name}\n")  # Dynamic job name
        f.write(SBATCH_TEMPLATE)
        f.write(f"CUDA_VISIBLE_DEVICES=0 python3 /scratchc/fmlab/zuberi01/phd/sequoia-pub/src/main.py \\\n")
        f.write(f"    --model_type vis \\\n")
        f.write(f"    --ref_file /scratchc/fmlab/zuberi01/phd/sequoia-pub/examples/matching_rows_sequoia.csv \\\n")
        f.write(f"    --save_dir output \\\n")
        f.write(f"    --cohort TCGA \\\n")
        f.write(f"    --exp_name {exp_name} \\\n")
        f.write(f"    --batch_size 16 \\\n")
        f.write(f"    --train \\\n")
        f.write(f"    --log histo_to_cnv \\\n")
        f.write(f"    --save_on loss+corr \\\n")
        f.write(f"    --stop_on loss+corr \\\n")
        f.write(f"    --split_column {split_column} \\\n")
        f.write(f"    --filter_no_features 0 \\\n")
        f.write(f"    --feature_path /scratchc/fmlab/zuberi01/masters/saved_patches/40x_400/features2 \\\n")
        f.write(f"    --rna_prefix cnv_ \n")

    
    # Submit the job
    os.system(f"sbatch {job_script}")
