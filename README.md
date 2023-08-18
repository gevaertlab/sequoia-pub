# Genes/Pathway prediction from H&E

## Folders

- The code is contained in the *src* folder. 
- In the *ref_files* folder, CSV are inside which have one row per WSI and their according gene expression values and labels. The RNA columns are named with the following format 'rna_GENNAME'. The 'tcga_project' column says the TCGA project that data belongs to.
- *config* folder contains the config files for the experiments. The majority of the content is depracated, but it is left there for legacy issues.
- *features* folder, where the features obtained from the WSI using the Resnet and the K-means algorithm later on are stored. They are saved in HDF5 files, and inside there are two datasets: **resnet_features** and **cluster_features**.

## GTEx conversion

To convert the GTEX data (obtain patches and then select the gene expression) there are two scripts ion the *GTex/preprocess/* folder:

Obtaining the patches:

```bash
python3 patch_gen_grid.py --wsi_path ../Histology/Colon/ --patch_path ../Histology/Colon_Patches256x256 --patch_size 256 --mask_path ../Histology/Colon_Patches_Masks --max_patches_per_slide 8000
```

Obtaining the genes:

```bash
python3 select_gene_expression.py --data_ref ../GTex_data_ref_new.csv --gene_file ../GTEx_genes.csv --tissue Pancreas --chunk_size 50
```

Once you have obtained the genes file, obtain only the proteincoding ones:

```bash
python3 select_proteincoding_genes.py --data_file GTex_GE_Liver.csv --protein_coding protein_coding_genes.csv --tissue Liver
```

Postprocess the csv to change wsi_file_names, add the rna_ prefix to the columns and create the new tcga_project column:

```bash
python3 postprocess_csv.py --csv_file GTex_Ovary_proteincoding.csv --gtex_tissue GTex_Ovary
```
## Execution

The majority of the scrips are saved in the *src/scripts/* folder once I am done with them, so there you can find all the examples.

### Step 1: Obtain resnet features

To obtain the resnet features we have two scripts, one for GTEx and one for TCGA data. The scripts are: *compute_resnet_features.py*, *compute_resnet_features_gtex.py*.

For instance, to compute the resnet features for TCGA-COAD it would be:

```bash
python3 compute_resnet_features.py --config ../config/config_coad_2000genes.json --start 0 --end 50
```

The ```--start``` and ```--end```, indicates the rows of the dataframe (storesd in the folder *ref_files*, the file is indicated in the config file), that are going to be converted. This is useful to execute the script in parallel to obtain the features, since the process is pretty slow.

Then, for the GTEx case, the script execution is pretty similar:

```bash
python3 compute_resnet_features_gtex.py --config ../config/config_gtex_brain.json --tissue GTex-Brain --start 112 --end 142
```

In this case you need to add the ```--tissue`` option to indicate the name of the folder where the features are going to be saved.

### Step 2: Obtain k-Means features

The next step once the resnet features have been obtained is to compute the 100 clusters used as input for the vision transformer. They are computed per slide, so it is pretty straightforward, and it is pretty fast. An example execution would be:

```bash
python3 kmean_features.py --config ../config/config_coad_2000genes.json --num_clusters 100 --tcga
```

The ```---tcga``` is to indicate that it is TCGA data. If it was GTEx data it would be:

```bash
python3 kmean_features.py --config ../config/config_gtex_brain.json --num_clusters 100 --gtex --gtex_tissue GTex-Brain
```

Again, you indicate the folder with the features with the ```--gtex_tissue``` parameter.

### Step 3: pretrain models on the GTEx data

We perform various experiments by pretraining first the vision transformer on the GTEx data. To do so you run:

```python
python3 pretrain_gtex.py --config ../config/config_gtex_lung.json --folder_feat GTex-Lung --exp_name gtex_lung_pretrain
```

The folder_feat parameter is only used if the the columnb *tcga_project* doesn't exists on the CSV. I know that the name is misleading since this is TCGA data, but I needed to go fast and I already had the code for TCGA so I didn't want to rename it or add a new parameter, sorry :(

For instance, when I am doing a multi-class pre-training, I added the folder manually to the CSV (e.g. 'gtex-brain-lung-proteincoding.csv')

### Step 4: Train the vision transformer on the TCGA data

Now we can train the vision transformer (or fine-tune it) on the TCGA data as:

```bash
python3 main.py --config ../config/config_kirp_2000genes.json --tcga_projects TCGA-KIRP --exp_name vit_kirp_kfold --batch_size 16 --k_fold --k 5 --train --num_genes 19198 --baseline
```

I think the parameters are pretty self-explanatory. The --num_genes indicates the number of genes, even though that is depracated now, since I compute it directly based on the dataframe. The --baseline is to also compute the baseline (use as prediction the mean of the genes). And the ```--train``` parameter is to train the model. If we want to use the pretrained weights (or any weights in general):

```bash
python3 main.py --config ../config/config_kirp_2000genes.json --tcga_projects TCGA-KIRP --exp_name vit_kirp_brainlungpretrainall_kfold --batch_size 16 --use_pretrain --checkpoint vit_exp/gtex_brainlung_pretrain_all/model_best.pt --k_fold --k 5 --train --num_genes 19198 --baseline
```

You just need to add the ```--use_pretrain``` and ```--checkpoint``` arguments. 

### Optional step: running the Owkin model

To run the Owkin model you just need to do:

```bash
python3 he2rna.py --config ../config/config_coad_2000genes.json --batch_size 64 --exp_name owkin_coad
```
