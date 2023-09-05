<p align="center">
  <img src="https://github.com/gevaertlab/sequoia-pub/blob/master/sequoia-logo.png"/>
</p>


# :evergreen_tree: SEQUOIA: Digital profiling of cancer transcriptomes with grouped vision attention

## Fold structure

- `scripts`: example bash (driver) scripts to run the pre-processing, training and evaluation.
- `examples`: example input files.
- `pre-processing`: pre-processing scripts.
- `evaluation`: evaluation scripts.
- `spatial_vis`: scripts for generating spatial predictions of gene expression values. 

## Pre-processing

Scripts for pre-processing are located in the `pre-processing` folder. All computational processes requires a *reference.csv* file, which has one row per WSI and their corresponding gene expression values. The RNA columns are named with the following format 'rna_{GENENAME}'. An optional 'tcga_project' column indicates the TCGA project that data belongs to. See `examples/ref_file.csv` for an example. 

### Step 1: Patch extraction

To extract patches from whole-slide images (WSIs), please use the script `patch_gen_hdf5.py`. 
An example script to run the patch extraction: `scripts/extract_patch.sh`

Note, the ```--start``` and ```--end``` parameters indicate the rows (WSIs) in the *reference.csv* file that need to be extracted. This is useful to execute the script in parallel.

### Step 2: Obtain resnet features

To obtain resnet features from patches, please use the script `compute_resnet_features_hdf5.py`. The script converts each patch into a linear feature vector. 

An example script to run the patch extraction: `scripts/extract_resnet_features.sh`

### Step 3: Obtain k-Means features

The next step once the resnet features have been obtained is to compute the 100 clusters used as input for the vision transformer. They are computed per slide, so it is pretty straightforward, and it is pretty fast. 

An example script to run the patch extraction: `scripts/extract_kmean_features.sh`

- Outputs from Step 2 and Step 3:
*features* folder, where the features obtained from the WSI using the Resnet and the K-means algorithm later on are stored. They are saved in HDF5 files, and inside there are two datasets: **resnet_features** and **cluster_features**.

## Pre-training and fine-tunning

### Step 4 (Optional): pretrain models on the GTEx data

To pretrain the weights of the model on normal tissues, please use the script `pretrain_gtex.py`. The process requires an input  *reference.csv* file, indicating the gene expression values for each WSI. See `examples/ref_file.csv` for an example. 

### Step 5: Train or fine-tune the vision transformer on the TCGA data

Now we can train the model from scratch or fine-tune it on the TCGA data. Here is an example bash script to run the process: `scripts/run_train.sh`

The parameters are explained within the `main.py` file. The ```--num_genes``` indicates the number of genes used for pretraining, which is depracated now. And the ```--train``` parameter is to train the model. To start from the pretrained weights, use the ```--use_pretrain``` and ```--checkpoint``` parameters. 

## Evaluation

Pearson correlation analysis is performed to compare the predicted gene expression values to ground truth. The significantly well predicted genes are selected using correlation coefficient, p value, and by statistical comparisons to an untrained model with the same architecture.

Evaluation script: `evaluation/vit_exp_corstats_TCGA.py`

## Benchmark with the HE2RNA model

To run the HE2RNA model, please use the bash script: `scripts/run_he2rna.sbatch`

## Spatial gene expression predictions

Scripts for predicting spatial gene expression levels within the same tissue slide are wrapped in: `spatial_vis`






