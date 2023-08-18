import os
import argparse
from tqdm import tqdm
import pickle

import h5py
from torch.utils.data import DataLoader, WeightedRandomSampler
import wandb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import torch.nn as nn
import random

from read_data import *
from utils import patient_kfold, grouped_strat_split
from vit import ViT, evaluate
from vit_new import train

def custom_collate_fn(batch):
    """Remove bad entries from the dataloader
    Args:
        batch (torch.Tensor): batch of tensors from the dataaset
    Returns:
        collate: Default collage for the dataloader
    """
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def filter_no_features(df, feature_path):
    projects = np.unique(df.tcga_project)
    all_wsis_with_features = []
    remove = []

    for proj in projects:
        wsis_with_features = os.listdir(feature_path + proj)
        # filter the ones without cluster_features
        for wsi in wsis_with_features:
            try:
                with h5py.File(feature_path +proj+ '/'+wsi+'/'+wsi+'.h5', "r") as f:
                    cols = list(f.keys())
                    if 'cluster_features' not in cols:
                        remove.append(wsi)
            except Exception as e:
                remove.append(wsi)
                
        all_wsis_with_features += wsis_with_features
    
    remove += df[~df['wsi_file_name'].isin(all_wsis_with_features)].wsi_file_name.values.tolist()
    print(f'Original shape: {df.shape}')
    df = df[~df['wsi_file_name'].isin(remove)].reset_index(drop=True)
    print(f'New shape: {df.shape}')

    return df

def filter_genes(data, selected_genes):
    selected_genes = [f'rna_{g}' for g in selected_genes]
    retain_cols = []
    for col in data.columns:
        if 'rna_' not in col:
            retain_cols.append(col)
            continue
        if 'rna_' in col and col in selected_genes:
            retain_cols.append(col)
    f_data = data.loc[:, retain_cols]
    return f_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Getting features')

    # general args
    parser.add_argument('--src_path', type=str, default='/oak/stanford/groups/ogevaert/data/Gen-Pred/', help='project path')
    parser.add_argument('--ref_file', type=str, default=None, help='path to reference file')
    parser.add_argument('--prediction_target', type=str, default='gene_expression', help='whether you are predicting gene expression or cell_state distr. should be "gene_expression" or "cell_state"')
    parser.add_argument('--tcga_projects', help="the tcga_projects we want to use, separated by comma", default=None, type=str)
    parser.add_argument('--feature_path', type=str, default="/oak/stanford/groups/ogevaert/data/Gen-Pred/features/", help='path to resnet and clustered features')
    parser.add_argument('--save_dir', type=str, default='vit_exp', help='parent destination folder')
    parser.add_argument('--cohort', type=str, default="TCGA", help='cohort name for creating the saving folder of the results')
    parser.add_argument('--exp_name', type=str, default="exp", help='Experiment name for creating the saving folder of the results')
    parser.add_argument('--log', type=int, default=1, help='Whether to log the loss during training')

    # model args
    parser.add_argument('--seed', type=int, default=99, help='Seed for random generation')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint from trained model. If tcga_pretrain is true, then the model paths in different folds should be of the format args.checkpoint + "{fold}.pt" ')
    parser.add_argument('--train', help="if you want to train the model", action="store_true")
    parser.add_argument('--baseline', help="computing the baseline", action="store_true")
    parser.add_argument('--filter_genes', type=str, default=None, help='path to a npy file containing a list of genes of interest for training')
    parser.add_argument('--change_num_genes', help="whether finetuning from a model trained on different number of genes", action="store_true")
    parser.add_argument('--num_genes', type=int, default=None, help='number of genes on which pretrained model was trained')
    parser.add_argument('--k', type=int, default=3, help='Number of splits')
    parser.add_argument('--tcga_pretrain', help="whether used pretrain model is pretrained on all TCGA cancers", action="store_true")
    parser.add_argument('--stratify', help="stratify k-fold for cancer type, only relevant if training on multiple cancer types", action="store_true")
    parser.add_argument('--balanced_sampling', help="balance sampling in dataloader according to number of available samples per cancer type, only relevant if training on multiple cancer types", action="store_true")
    parser.add_argument('--save_on', type=str, default='loss', help='which criterium to save model on, "loss" or "loss+corr"')
    parser.add_argument('--stop_on', type=str, default='loss', help='which criterium to do early stopping on, "loss" or "loss+corr"')

    ############################################## variables ##############################################
    args = parser.parse_args()

    ############################################## seeds ##############################################
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    #os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2" # needed for torch.use_determ (below) --> still throws the same error
    torch.backends.cudnn.benchmark = False # possibly reduced performance but better reproducibility
    torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)

    # reproducibility train dataloader
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(0)

    ############################################## logging ##############################################
    save_dir = os.path.join(args.src_path, args.save_dir, args.cohort, args.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    run = None
    if args.log:
        run = wandb.init(project="visgene", entity='mpizuric', config=args, name=args.exp_name) 

    ############################################## data prep ##############################################
    path_csv = args.ref_file
    df = pd.read_csv(path_csv)

    # filter tcga projects
    if ('tcga_project' in df.columns) and (args.tcga_projects != None):
        projects = args.tcga_projects.split(',')
        df = df[df['tcga_project'].isin(projects)].reset_index(drop=True)

    print("filtering slides whose cluster features do not exist")
    df = filter_no_features(df, args.feature_path)

    # filter genes of interest
    selected_genes = None
    if args.filter_genes is not None:
        selected_genes = np.load(args.filter_genes, allow_pickle=True)
        selected_genes = list(set(selected_genes))
        df = filter_genes(df, selected_genes)
        print(f"Training only for selected genes: n = {len(selected_genes)}")

    ############################################## train, val, test split ##############################################
    train_idxs, val_idxs, test_idxs = patient_kfold(df, n_splits=args.k)

    ############################################## kfold ##############################################
    test_results_splits = {}
    i = 0
    for train_idx, val_idx, test_idx in zip(train_idxs, val_idxs, test_idxs):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        test_df = df.iloc[test_idx]
        
        # save patient ids to file
        np.save(save_dir + '/train_'+str(i)+'.npy', np.unique(train_df.patient_id) )
        np.save(save_dir + '/val_'+str(i)+'.npy', np.unique(val_df.patient_id) )
        np.save(save_dir + '/test_'+str(i)+'.npy', np.unique(test_df.patient_id) )

        if args.prediction_target == "gene_expression":
            train_dataset = SuperTileRNADataset(train_df, args.feature_path)
            val_dataset = SuperTileRNADataset(val_df, args.feature_path)
            test_dataset = SuperTileRNADataset(test_df, args.feature_path)

            if args.baseline:
                rna_columns = [x for x in train_df.columns if 'rna_' in x]
                rna_values = train_df[rna_columns].values
                mean_baseline = np.mean(rna_values, axis=0)

        elif args.prediction_target == "cell_state":
            train_dataset = SuperTileCellStatesDataset(train_df, args.feature_path)
            val_dataset = SuperTileCellStatesDataset(val_df, args.feature_path)
            test_dataset = SuperTileCellStatesDataset(test_df, args.feature_path)
        
        else:
            print("please provide correct prediction target")
            exit()

        if args.balanced_sampling:
            weights = pd.DataFrame(train_df.shape[0]/train_df.tcga_project.value_counts()).reset_index()
            weights.columns = ['tcga_project', 'weight']
            train_df_weights = weights.merge(train_df, on='tcga_project').head()
            samples_weight = torch.from_numpy(train_df_weights.weight.values).double()
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            shuffle = False # mutually exclusive with sampler
        else:
            sampler = None
            shuffle = True

        train_dataloader = DataLoader(train_dataset, 
                    num_workers=0, pin_memory=True, 
                    shuffle=shuffle, batch_size=args.batch_size,
                    collate_fn=custom_collate_fn, sampler=sampler,
                    worker_init_fn=seed_worker,
                    generator=g)
        
        val_dataloader = DataLoader(val_dataset, 
                    num_workers=0, pin_memory=True, 
                    shuffle=True, batch_size=args.batch_size,
                    collate_fn=custom_collate_fn)
        
        test_dataloader = DataLoader(test_dataset, 
                    num_workers=0, pin_memory=True, 
                    shuffle=False, batch_size=args.batch_size,
                    collate_fn=custom_collate_fn)
        
        num_outputs = train_dataset.num_genes if args.prediction_target == "gene_expression" else train_dataset.num_cell_states
        if args.checkpoint and args.change_num_genes: # if finetuning from model trained on gtex/tcga
            model_path = args.checkpoint 
            if args.tcga_pretrain: # then we need to load different checkpoints for different folds (if pretrained on gtex, then just use the best model from gtex for all folds)
                model_path = model_path + str(i) + '.pt'

            model = ViT(num_outputs=args.num_genes, 
                    dim=2048, depth=6, heads=16, mlp_dim=2048, dim_head = 64) 
            model.load_state_dict(torch.load(model_path))
            
            model.linear_head = nn.Sequential(
                nn.LayerNorm(2048),
                nn.Linear(2048, num_outputs))

        else: # if training from scratch or continuing training same model (then load state dict in next if)
            model = ViT(num_outputs=num_outputs, 
                    dim=2048, depth=6, heads=16, mlp_dim=2048, dim_head = 64) 
        
        if args.checkpoint and not args.change_num_genes:
            model.load_state_dict(torch.load(args.checkpoint))
        
        model = model.cuda()

        optimizer = torch.optim.AdamW(list(model.parameters()), 
                                        lr=args.lr, 
                                        amsgrad=False,
                                        weight_decay=0.)

        dataloaders = {
            'train': train_dataloader,
            'val': val_dataloader
        }
        
        if args.train:
            model = train(model, dataloaders, optimizer, save_dir=save_dir, run=run, split=i, save_on=args.save_on, stop_on=args.stop_on, delta=0.5)

        preds, real, wsis, projs = evaluate(model, test_dataloader, run=run, suff='_'+str(i))

        random_model = ViT(num_outputs=num_outputs, dim=2048, depth=6, heads=16, mlp_dim=2048, dim_head = 64)  
        random_model = random_model.cuda()
        random_preds, _, _, _ = evaluate(random_model, test_dataloader, run=run, suff='_'+str(i)+'_rand')
        
        if args.baseline and (args.prediction_target == "gene_expression"):
            mean_baseline = mean_baseline.reshape(-1, mean_baseline.shape[0])
            mean_baseline = np.repeat(mean_baseline, real.shape[0], axis=0)
            mse = mean_squared_error(real, mean_baseline)
            mae = mean_absolute_error(real, mean_baseline)
            print(f'Baseline test MSE {mse}')
            print(f'Baseline test MAE {mae}')

        test_results = {
            'real': real,
            'preds': preds,
            'random': random_preds,
            'wsi_file_name': wsis,
            'tcga_project': projs,
            'genes':[x for x in df.columns if 'rna_' in x]
        }

        if args.baseline and (args.prediction_target == "gene_expression"):
            test_results['baseline'] = mean_baseline
        
        test_results_splits[f'split_{i}'] = test_results
        i += 1
    
    test_results_splits['genes'] = [x[4:] for x in df.columns if 'rna_' in x]
    with open(os.path.join(save_dir, 'test_results.pkl'), 'wb') as f:
        pickle.dump(test_results_splits, f, protocol=pickle.HIGHEST_PROTOCOL)