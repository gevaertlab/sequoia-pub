<p align="center">
  <img src="https://github.com/gevaertlab/sequoia-pub/blob/master/images/seq-logo.png"/>
</p>


# :evergreen_tree: SEQUOIA: Digital profiling of cancer transcriptomes with grouped vision attention

**Abstract**

_Cancer is a heterogeneous disease that demands precise molecular profiling for better understanding and management. RNA-sequencing has emerged as a potent tool to unravel the moleclular heterogeneity. However, large-scale characterization of cancer transcriptomes is hindered by the limitations of costs and tissue accessibility. Here, we develop SEQUOIA, a deep learning model employing a transformer encoder to predict cancer transcriptomes from whole-slide histology images. We pre-train the model using data from 2,242 normal tissues, and the model is fine-tuned and evaluated in 4,218 tumor samples across nine cancer types. The model is further validated across two independent cohorts compromising 1,305 tumors. The highest performance was observed in cancers from breast, kidney and lung, where  SEQUOIA accurately predicted 13,798, 10,922 and 9,735 genes, respectively. The well predicted genes are associated with the regulation of inflammatory response, cell cycles and hypoxia-related metabolic pathways. Leveraging the well predicted genes, we develop a digital signature to predict the risk of recurrence in breast cancer. While the model is trained at the tissue-level, we showcase its potential in predicting spatial gene expression patterns using spatial transcriptomics datasets. SEQUOIA deciphers the molecular complexity of tumors from histology images, opening avenues for improved cancer management and personalized therapies._

**Overview**
<p align="center">
  <img src="https://github.com/gevaertlab/sequoia-pub/blob/master/images/overview2.png"/>
</p>

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

The parameters are explained within the `main.py` file. The ```--num_genes``` indicates the number of genes that were used for pretraining (needed for checkpoint loading). And the ```--train``` parameter is to train the model. To start from the pretrained weights, use the ```--use_pretrain``` and ```--checkpoint``` parameters. 

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



