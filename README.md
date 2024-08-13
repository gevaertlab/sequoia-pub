<p align="center">
  <img src="https://github.com/gevaertlab/sequoia-pub/blob/master/images/seq-logo.png"/>
</p>


# :evergreen_tree: SEQUOIA: Digital profiling of cancer transcriptomes with grouped vision attention

**Abstract**

_Cancer is a heterogeneous disease requiring precise yet costly genetic profiling for better understanding and management. Recently, deep learning models have demonstrated potential for cost-efficient prediction of genetic alterations from whole slide histology images (WSIs). While transformer architectures have enabled significant progress in non-medical domains, their application to WSIs lags behind due to small dataset sizes coupled with the explosion of trainable parameters. Here, we develop SEQUOIA, a customized transformer model with a linear-complexity alternative to self-attention, for better gene expression prediction from WSIs. To further mitigate small dataset sizes, we leverage an advanced large-scale self-supervised model (UNI) for WSI feature extraction. We develop and evaluate SEQUOIA in 7,584 tumor samples across sixteen cancer types. The prediction performance is assessed both at individual gene levels and at pathway levels through Pearson correlation analysis and root mean squared error. Generalization is validated on two independent cohorts with 1,368 tumors. Across 20,820 genes, the highest performance is observed breast, kidney and lung cancer, where SEQUOIA accurately predicts the expression of 18,977, 17,922 and 17,354 genes, respectively. The accurately predicted genes are associated with key cancer processes including the regulation of inflammatory response, cell cycles and metabolisms. Leveraging the predictions, we develop a digital gene expression signature that predicts the risk of recurrence in breast cancer. While the model is trained at the tissue level, we showcase its potential in predicting spatial gene expression patterns using two spatial transcriptomics datasets. SEQUOIA hence deciphers clinically relevant gene expression patterns from WSIs, opening avenues for improved cancer management and personalized therapies._

**Overview**
<p align="center">
  <img src="https://github.com/gevaertlab/sequoia-pub/blob/master/images/overview_new.png"/>
</p>

## Fold structure

- `scripts`: example bash (driver) scripts to run the pre-processing, training and evaluation.
- `examples`: example input files.
- `pre-processing`: pre-processing scripts.
- `evaluation`: evaluation scripts.
- `spatial_vis`: scripts for generating spatial predictions of gene expression values. 
- `src`: main files for models and training.

## System requirements

Software dependencies and versions are listed in requirements.txt

## Installation

First, clone this git repository: `git clone https://github.com/gevaertlab/sequoia-pub.git`

Then, create a conda environment: `conda create -n sequoia python=3.9` and activate: `conda activate sequoia`

Install the openslide library: `conda install -c conda-forge openslide==4.0.0`

Install the required package dependencies: `pip install -r requirements.txt`

Finally, install [Openslide](https://openslide.org/download/) (>v3.4.0)

Expected installation time in normal Linux environment: 15 mins 

## Pre-processing

Scripts for pre-processing are located in the `pre-processing` folder. All computational processes requires a *reference.csv* file, which has one row per WSI and their corresponding gene expression values. The RNA columns are named with the following format 'rna_{GENENAME}'. An optional 'tcga_project' column indicates the TCGA project that data belongs to. See `examples/ref_file.csv` for an example. 

### Step 1: Patch extraction

To extract patches from whole-slide images (WSIs), please use the script `patch_gen_hdf5.py`. 
An example script to run the patch extraction: `scripts/extract_patch.sh`

Note, the ```--start``` and ```--end``` parameters indicate the rows (WSIs) in the *reference.csv* file that need to be extracted. This is useful to execute the script in parallel.

### Step 2: Obtain resnet/uni features

To obtain resnet/uni features from patches, please use the script `compute_features_hdf5.py`. The script converts each patch into a linear feature vector. 

Note: if you use the UNI model, you need to follow the installation procedure in the original [github](https://github.com/mahmoodlab/UNI) and install the necessary [required packages](https://github.com/mahmoodlab/UNI/blob/main/setup.py).

An example script to run the patch extraction: `scripts/extract_resnet_features.sh`

### Step 3: Obtain k-Means features

The next step once the resnet features have been obtained is to compute the 100 clusters used as input for the vision transformer. They are computed per slide, so it is pretty straightforward, and it is pretty fast. 

An example script to run the patch extraction: `scripts/extract_kmean_features.sh`

- Outputs from Step 2 and Step 3:
*features* folder, where the features obtained from the WSI using the Resnet and the K-means algorithm later on are stored. They are saved in HDF5 files, and inside there are two datasets: **resnet_features** and **cluster_features**.

Expected run time: depend on the hardware (CPU/GPU) and the number of slides

## Pre-training and fine-tunning

### Step 4 (Optional): pretrain models on the GTEx data

To pretrain the weights of the model on normal tissues, please use the script `pretrain_gtex.py`. The process requires an input  *reference.csv* file, indicating the gene expression values for each WSI. See `examples/ref_file.csv` for an example. 

### Step 5: Train or fine-tune the vision transformer on the TCGA data

Now we can train the model from scratch or fine-tune it on the TCGA data. Here is an example bash script to run the process: `scripts/run_train.sh`

The parameters are explained within the `main.py` file. The ```--num_genes``` indicates the number of genes that were used for pretraining (needed for checkpoint loading). And the ```--train``` parameter is to train the model. To start from the pretrained weights, use the ```--use_pretrain``` and ```--checkpoint``` parameters. 

Expected run time: depend on the hardware (CPU/GPU) and the number of slides

## Evaluation

Pearson correlation analysis is performed to compare the predicted gene expression values to ground truth. The significantly well predicted genes are selected using correlation coefficient, p value, and by statistical comparisons to an untrained model with the same architecture.

Evaluation script: `evaluation/vit_exp_corstats_TCGA.py`

## Benchmark with the HE2RNA model

To run the HE2RNA model, please use the bash script: `scripts/run_he2rna.sbatch`

## Spatial gene expression predictions

Scripts for predicting spatial gene expression levels within the same tissue slide are wrapped in: `spatial_vis`

- ```visualize.py``` is the file to generate spatial predictions made with a saved SEQUOIA model. 
  - the arguments are explained in the file
  - output: the output is a dataframe that contains the following columns:
  ```
  - xcoord: the x coordinate of a tile (absolute position of tile in the WSI -- note that adjacent tiles will have coordinates that are tile_width apart!)
  - ycoord: same as xcoord for the y
  - xcoord_tf: the x coordinate of a tile when transforming the original coordinates to start in the left upper corner at position x=0,y=0 and with distance 1 between tiles (i.e. next tile has coordinate x=1,y=0)
  - ycoord_tf: same as xcoord_tf for the y
  - gene_{x}: for each gene, there will be a column 'gene_{x}' that contains the spatial prediction for that gene of the model from fold {x}, with x = 1..num_folds
  - gene: for each gene there will also be a column without the _{x} part, which represents the average across the used folds
  ```
- ```get_emd.py``` contains code to calculate the two dimensional Earth Mover's Distance between a prediction map (generated with ```visualize.py``` script) and ground truth spatial transcriptomics.
- ```gbm_celltype_analysis.py``` contains (1) code to examine spatial co-expression of genes for the four meta-modules described in the paper; (2) code to visualize spatial organization of meta-modules on the considered slides.


# License

&copy; [Gevaert's Lab](https://med.stanford.edu/gevaertlab.html) MIT License



