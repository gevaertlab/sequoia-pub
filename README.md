<p align="center">
  <img src="https://github.com/gevaertlab/sequoia-pub/blob/master/images/seq-logo.png"/>
</p>


# :evergreen_tree: SEQUOIA: Digital profiling of cancer transcriptomes with linearized attention

**Abstract**

_Cancer is a heterogeneous disease requiring costly genetic profiling for better understanding and management. Recent advances in deep learning have enabled cost-effective predictions of genetic alterations from whole slide images (WSIs). While transformers have driven significant progress in non-medical domains, their application to WSIs lags behind due to high model complexity and limited dataset sizes. Here, we introduce SEQUOIA, a linearized transformer model that predicts cancer transcriptomic profiles from WSIs. SEQUOIA is developed using 7,584 tumor samples across 16 cancer types, with its generalization capacity validated on two independent cohorts comprising 1,368 tumors. Accurately predicted genes are associated with key cancer processes, including inflammatory response, cell cycles and metabolism. Further, we demonstrate the value of SEQUOIA in stratifying the risk of breast cancer recurrence and in resolving spatial gene expression at loco-regional levels. SEQUOIA hence deciphers clinically relevant information from WSIs, opening avenues for personalized cancer management._

**Overview**
<p align="center">
  <img src="https://github.com/gevaertlab/sequoia-pub/blob/master/images/overview_new.png"/>
</p>

## Folder structure

- `scripts`: example bash (driver) scripts to run the pre-processing, training and evaluation.
- `examples`: example input files.
- `pre-processing`: pre-processing scripts.
- `evaluation`: evaluation scripts and output gene list ordered by index. 
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

The next step once the resnet/uni features have been obtained is to compute the 100 clusters used as input for the model. They are computed per slide, so it is pretty straightforward, and it is pretty fast. 

An example script to run the patch extraction: `scripts/extract_kmean_features.sh`

- Outputs from Step 2 and Step 3:
*features* folder, this contains for each WSI a HDF5 file that stores both the features obtained using the resnet/uni (inside the **resnet_features** or **uni_features** dataset) as well as the output from the K-means algorithm (inside **cluster_features** dataset).

Expected run time: depend on the hardware (CPU/GPU) and the number of slides

## Running evaluation of our pre-trained model on an independent dataset

After WSI pre-processing, our pre-trained SEQUOIA model (UNI features and linearized transformer aggregation) can be evaluated on the WSIs of an independent dataset by running ``evaluation/predict_independent_dataset.py``. 

We released the weights for each cancer type, from each of the five folds on [HuggingFace](https://huggingface.co/gevaertlab), so make sure to login (See [HuggingFace docs](https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command) for more information):

```
from huggingface_hub import login
login()
```

The gene names corresponding to the output can be found in the `evaluation/gene_list.csv` file. 

## Pre-training, fine-tunning and loading pre-trained weights

### Step 1 (Optional): pretrain models on the GTEx data

To pretrain the weights of the model on normal tissues, please use the script `pretrain_gtex.py`. The process requires an input  *reference.csv* file, indicating the gene expression values for each WSI. See `examples/ref_file.csv` for an example. 

### Step 2 (Optional): load the same train/validation/test splits that we used

The TCGA splits for each fold are available in the `patient_splits.zip` file in the [pre_processing](https://github.com/gevaertlab/sequoia-pub/blob/master/pre_processing/patient_splits.zip) folder. 

To load the splits from the numpy file, unzip the `patient_splits.zip` folder. To use:

```
split = np.load(f'TCGA-{cancer}.npy'), allow_pickle=True).item()
for i in range(5):
  train_patients = split[f'fold_{i}']['train']
  val_patients = split[f'fold_{i}']['val']
  test_patients = split[f'fold_{i}']['test']
```

Note that these contain only the patient ID, not the entire WSI filename. The WSI file names within each test fold are available in `test_wsis.pkl` in the same [pre_processing](https://github.com/gevaertlab/sequoia-pub/blob/master/pre_processing/) folder. To use:

```
with open('test_wsis.pkl','rb') as f:
  data = pickle.load(f)
test_wsis = data[f'{cancer}']['split_{i}']
```

Concatenating all the WSIs from a particular cancer type across all the folds results in all the WSI IDs that were used for that cancer type. So to find the exact WSI filenames used in the train/validation split from fold 0, match the patient IDs from `train_patients` and `val_patients` above to the WSI IDs across folds 1-4 in `test_wsis.pkl`:

```
train_patients = split['fold_0']['train']
val_patients = split['fold_0']['val']
wsis = np.concatenate([data['brca'][f'split_{i}']['wsi_file_name'] for i in range(1,5)])
train_wsis = [i for i in wsis if '-'.join(i.split('-')[:3]) in train_patients]
val_wsis = [i for i in wsis if '-'.join(i.split('-')[:3]) in val_patients]

```

### Step 3 (Optional): load published model checkpoint

As mentioned above, our pre-trained checkpoint weights for SEQUOIA are available on [HuggingFace](https://huggingface.co/gevaertlab). Patients that were present in the test set in each fold can be found in `src/folds`. Make sure to login to HuggingFace (see above).

Then use:
```
from src.tformer_lin import ViS

cancer = 'brca'
i = 0 ## fold number
model = ViS.from_pretrained(f"gevaertlab/sequoia-{cancer}-{i}")
```
The gene names corresponding to the output can be found in the `evaluation/gene_list.csv` file. 

### Step 4: Train or fine-tune SEQUOIA on the TCGA data

Now we can train the model from scratch or fine-tune it on the TCGA data. Here is an example bash script to run the process: `scripts/run_train.sh`

The parameters are explained within the `main.py` file. 

Some points that we want to emphasize:
- If you pre-trained on a dataset that contains a different number of genes than the finetuning dataset, you need to set the ```--change_num_genes``` parameter to 1 and specify in the ```--num_genes``` parameter how many genes were used for pretraining. To indicate the path to the pretrained weights, use the ```--checkpoint``` parameters. 
- ```--model_type``` is used to define the aggregation type. For the SEQUOIA model (linearized transformer aggregation) use 'vis'. 

## Benchmarking

For running the benchmarked variations of the architecture:
- MLP aggregation: for this part we made use of the implementation from HE2RNA, which can be found in `he2rna.py`. An example run script is provided in `scripts/run_he2rna.sh`
- transformer aggregation: this model type is implemented in the `main.py`. use ```--model_type``` 'vit'.


## Evaluation

Pearson correlation and RMSE values are calculated to compare the predicted gene expression values to the ground truth. The significantly well predicted genes are selected using correlation coefficient, p value, rmse, and by statistical comparisons to an untrained model with the same architecture.

Evaluation script: `evaluation/evaluate_model.py`. Output: three dataframes `all_genes.csv`: contains evaluation metrics for all genes, `sig_genes.csv`: metrics for only the significant genes and `num_sig_genes.csv` contains the number of significant genes per cancer type with this model.

## Spatial gene expression predictions

Scripts for predicting spatial gene expression levels within the same tissue slide are wrapped in: `spatial_vis`

- ```visualize.py``` is the file to generate spatial predictions made with a saved SEQUOIA model. 
  - the arguments are explained in the file. an example run file is provided in `scripts/run_visualize.sh`
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



