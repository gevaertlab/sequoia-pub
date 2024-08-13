import os
import json
import argparse
import pdb
from tqdm import tqdm
import datetime
import pickle

import h5py
from sklearn.cluster import KMeans
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import wandb
from accelerate import Accelerator
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import torch.nn as nn

from src.read_data import *
from src.utils import patient_kfold, grouped_strat_split
from src.vit import ViT, train, evaluate, predict
from src.he2rna import HE2RNA, he2rna_predict

def custom_collate_fn(batch):
    """Remove bad entries from the dataloader
    Args:
        batch (torch.Tensor): batch of tensors from the dataaset
    Returns:
        collate: Default collage for the dataloader
    """
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def filter_wsi_with_features(df, feature_path):
    # filter out WSIs for which we don't have features
    projects = np.unique(df.tcga_project)
    selected_wsi = []

    for proj in projects:
        wsis_with_features = os.listdir(os.path.join(feature_path, proj))
        # filter the ones without cluster_features
        for wsi in wsis_with_features:
            h5_path = os.path.join(feature_path, proj, wsi, f'{wsi}.h5')
            try:
                with h5py.File(h5_path, "r") as f:
                    cols = list(f.keys())
                    if 'cluster_features' in cols:
                        selected_wsi.append(wsi)
            except Exception as e:
                print(wsi)
                print(e)

    print(f'Original shape: {df.shape}')
    df = df[df['wsi_file_name'].isin(selected_wsi)].reset_index(drop=True)
    print(f'New shape: {df.shape}')
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Getting features')
    parser.add_argument('--ref_file', type=str, required=True, help='Reference file')
    parser.add_argument('--feature_path', type=str, default="/oak/stanford/groups/ogevaert/data/Gen-Pred/features/", help='Output directory to save features')
    parser.add_argument('--folds', type=int, default=1, help='Folds for pre-trained model')
    parser.add_argument('--seed', type=int, default=99, help='Seed for random generation')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--he2rna', type=int, default=0, help='whether to use the he2rna model')
    parser.add_argument('--num_clusters', type=int, default=100, help='Number of clusters for the kmeans')
    parser.add_argument('--tcga_projects', default=None, type=str, nargs='*', help="the tcga_projects we want to use")
    parser.add_argument('--save_dir', type=str, default="/oak/stanford/groups/ogevaert/data/Gen-Pred/vit_exp", help='save_path')
    parser.add_argument('--exp_name', type=str, default="exp", help='Experiment name')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint from trained model')

    ############################################## variables ##############################################
    args = parser.parse_args()

    ############################################## seeds ##############################################
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ############################################## logging ##############################################

    save_dir = os.path.join(args.save_dir, args.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    run = None
    #run = wandb.init(project="visgene", entity='mpizuric', config=args, name=args.exp_name)

    ############################################## data prep ##############################################
    path_csv = args.ref_file
    df = pd.read_csv(path_csv)

    # filter out WSIs for which we don't have features
    df = filter_wsi_with_features(df, args.feature_path)

    if 'tcga_project' in df.columns and args.tcga_projects:
        df = df[df['tcga_project'].isin(args.tcga_projects)].reset_index(drop=True)

    genes = [c[4:] for c in df.columns if "rna_" in c]

    test_dataset = SuperTileRNADataset(df, args.feature_path)

    test_dataloader = DataLoader(test_dataset,
                num_workers=0, pin_memory=True,
                shuffle=False, batch_size=args.batch_size,
                collate_fn=custom_collate_fn)

    res_preds = []
    res_random = []

    for fold in range(args.folds):
        print(f'Predicting fold: {fold}')

        if fold == 0:
            if os.path.exists(os.path.join(args.checkpoint, f'model_best.pt')):
                pretrained_model = os.path.join(args.checkpoint, f'model_best.pt')
            else:
                pretrained_model = os.path.join(args.checkpoint, f'model_{fold}.pt')
        else:
            if os.path.exists(os.path.join(args.checkpoint, f'model_best_{fold}.pt')):
                pretrained_model = os.path.join(args.checkpoint, f'model_best_{fold}.pt')
            else:
                pretrained_model = os.path.join(args.checkpoint, f'model_{fold}.pt')

        if args.he2rna:
            model = torch.load(pretrained_model)
            preds, labels, wsis, projs = he2rna_predict(model, test_dataloader)
            random_model = HE2RNA(input_dim=2048, layers=[256,256],
                    ks=[1,2,5,10,20,50,100],
                    output_dim=test_dataset.num_genes, device="cuda")
            random_preds, _, _, _ = he2rna_predict(random_model, test_dataloader)
        else:
            model = ViT(num_outputs=test_dataset.num_genes, dim=2048, depth=6, heads=16, mlp_dim=2048, dim_head = 64)
            model.load_state_dict(torch.load(pretrained_model))
            model = model.cuda()

            preds, wsis, projs = predict(model, test_dataloader, run=run)
            random_model = ViT(num_outputs=test_dataset.num_genes, dim=2048, depth=6, heads=16, mlp_dim=2048, dim_head = 64)
            random_model = random_model.cuda()
            random_preds, _, _ = predict(random_model, test_dataloader, run=run)

        res_preds.append(preds)
        res_random.append(random_preds)

    # calcualte average accross folds
    avg_preds = np.mean(res_preds, axis = 0)
    avg_random = np.mean(res_random, axis = 0)

    df_pred = pd.DataFrame(avg_preds, index = wsis, columns = genes)
    df_random = pd.DataFrame(avg_random, index = wsis, columns = genes)

    test_results = {'pred': df_pred, 'random': df_random}

    with open(os.path.join(save_dir, 'test_results.pkl'), 'wb') as f:
        pickle.dump(test_results, f, protocol=pickle.HIGHEST_PROTOCOL)
