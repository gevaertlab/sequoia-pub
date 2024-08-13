import os
import argparse
from tqdm import tqdm
import pickle
import h5py
import wandb
import random

import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from read_data import SuperTileRNADataset
from utils import patient_kfold, filter_no_features, custom_collate_fn
from vit import train, ViT, evaluate
from tformer_lin import ViS


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Getting features')

    # general args
    parser.add_argument('--src_path', type=str, default='', help='project path')
    parser.add_argument('--ref_file', type=str, default=None, help='path to reference file')
    parser.add_argument('--sample-percent', type=float, default=None, help='Downsample available data to test the effect of having a smaller dataset. If None, no downsampling.')
    parser.add_argument('--tcga_projects', help="the tcga_projects we want to use, separated by comma", default=None, type=str)
    parser.add_argument('--feature_path', type=str, default="features/", help='path to resnet and clustered features')
    parser.add_argument('--feature_use', type=str, default="cluster_features", help='which feature to use for training the model')
    parser.add_argument('--save_dir', type=str, default='vit_exp', help='parent destination folder')
    parser.add_argument('--cohort', type=str, default="TCGA", help='cohort name for creating the saving folder of the results')
    parser.add_argument('--exp_name', type=str, default="exp", help='Experiment name for creating the saving folder of the results')
    parser.add_argument('--filter_no_features', type=int, default=1, help='Whether to filter out samples with no features')
    parser.add_argument('--log', type=str, help='Experiment name to log')
    
    # model args
    parser.add_argument('--model_type', type=str, default='vit', help='"vit" for transformer or "vis" for linearized transformer')
    parser.add_argument('--depth', type=int, default=6, help='transformer depth')
    parser.add_argument('--num-heads', type=int, default=16, help='number of attention heads')
    parser.add_argument('--seed', type=int, default=99, help='Seed for random generation')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint from trained model.')
    parser.add_argument('--train', help="if you want to train the model", action="store_true")
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--change_num_genes', type=int, default=0, help="whether finetuning from a model trained on different number of genes")
    parser.add_argument('--num_genes', type=int, default=None, help='number of genes on which pretrained model was trained')
    parser.add_argument('--k', type=int, default=5, help='Number of splits')
    parser.add_argument('--save_on', type=str, default='loss', help='which criterium to save model on, "loss" or "loss+corr"')
    parser.add_argument('--stop_on', type=str, default='loss', help='which criterium to do early stopping on, "loss" or "loss+corr"')

    args = parser.parse_args()
    
    ############################################## seeds ##############################################
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False # possibly reduced performance but better reproducibility
    torch.backends.cudnn.deterministic = True

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
        run = wandb.init(project=args.log, config=args, name=args.exp_name) 
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)

    ############################################## data prep ##############################################
    
    df = pd.read_csv(args.ref_file)
    if args.sample_percent != None:
        df = df.sample(frac=args.sample_percent).reset_index(drop=True)

    if ('tcga_project' in df.columns) and (args.tcga_projects != None):
        projects = args.tcga_projects.split(',')
        df = df[df['tcga_project'].isin(projects)].reset_index(drop=True)
        print(f'Filtered project {projects}')

    if args.filter_no_features:
        df = filter_no_features(df, feature_path=args.feature_path, feature_name=args.feature_use)
    
    ############################################## kfold ##############################################
    train_idxs, val_idxs, test_idxs = patient_kfold(df, n_splits=args.k)

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
        
        # init dataset
        train_dataset = SuperTileRNADataset(train_df, args.feature_path, args.feature_use)
        val_dataset = SuperTileRNADataset(val_df, args.feature_path, args.feature_use)
        test_dataset = SuperTileRNADataset(test_df, args.feature_path, args.feature_use)

        num_outputs = train_dataset.num_genes 
        feature_dim = train_dataset.feature_dim

        # init dataloaders
        train_dataloader = DataLoader(train_dataset, 
                    num_workers=0, pin_memory=True, 
                    shuffle=shuffle, batch_size=args.batch_size,
                    collate_fn=custom_collate_fn,
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
        
        # get model
        if args.checkpoint and args.change_num_genes: # if finetuning from model trained on gtex
            model_path = os.path.join(args.checkpoint)
            if args.model_type == 'vit':
                model = ViT(num_outputs=args.change_num_genes, dim=feature_dim, 
                            depth=args.depth, heads=args.num_heads, 
                            mlp_dim=2048, dim_head=64, device=device) 
            elif args.model_type == 'vis':
                model = ViS(num_outputs=args.change_num_genes, input_dim=feature_dim,  
                            depth=args.depth, nheads=args.num_heads, 
                            dimensions_f=64, dimensions_c=64, dimensions_s=64, device=device)
            else:
                print('please specify correct model type "vit" or "vis"')
                exit()
                
            model.load_state_dict(torch.load(model_path, map_location = device))
            print(f'Loaded model from {model_path}')

            model.linear_head = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, num_outputs))

        else: # if training from scratch or continuing training same model (then load state dict in next if)
            if args.model_type == 'vit':
                model = ViT(num_outputs=num_outputs, dim=feature_dim, 
                            depth=args.depth, heads=args.num_heads, 
                            mlp_dim=2048, dim_head=64, device=device) 
            elif args.model_type == 'vis':
                model = ViS(num_outputs=num_outputs, input_dim=feature_dim, 
                            depth=args.depth, nheads=args.num_heads, 
                            dimensions_f=64, dimensions_c=64, dimensions_s=64, device=device)
            else:
                print('please specify correct model type "vit" or "vis"')

        if args.checkpoint and not args.change_num_genes:
            suff = f'_{i}' if i > 0 else ''
            model_path = args.checkpoint + f'model_best{suff}.pt'
            print(f'Loading model from {model_path}')
            model.load_state_dict(torch.load(model_path, map_location=device))

        model.to(device)

        # training 
        optimizer = torch.optim.AdamW(list(model.parameters()), 
                                        lr=args.lr, 
                                        amsgrad=False,
                                        weight_decay=0.)
        dataloaders = { 'train': train_dataloader, 'val': val_dataloader}
        
        if args.train:
            model = train(model, dataloaders, optimizer, 
                            num_epochs=args.num_epochs, run=run,split=i, 
                            save_on=args.save_on, stop_on=args.stop_on, 
                            delta=0.5, save_dir=save_dir)

        preds, real, wsis, projs = evaluate(model, test_dataloader, run=run, suff='_'+str(i))

        # get random model predictions
        if args.model_type == 'vit':
            random_model = ViT(num_outputs=num_outputs, dim=feature_dim, 
                                depth=args.depth, heads=args.num_heads, 
                                mlp_dim=2048, dim_head = 64, device = device)  
        else:
            random_model = ViS(num_outputs=num_outputs, input_dim=feature_dim, 
                            depth=args.depth, nheads=args.num_heads,  
                            dimensions_f=64, dimensions_c=64, dimensions_s=64, device=device)
        random_model = random_model.to(device)
        random_preds, _, _, _ = evaluate(random_model, test_dataloader, run=run, suff='_'+str(i)+'_rand')
        
        test_results = {
            'real': real,
            'preds': preds,
            'random': random_preds,
            'wsi_file_name': wsis,
            'tcga_project': projs
        }
        
        test_results_splits[f'split_{i}'] = test_results
        i += 1

    test_results_splits['genes'] = [x[4:] for x in df.columns if 'rna_' in x]
    with open(os.path.join(save_dir, 'test_results.pkl'), 'wb') as f:
        pickle.dump(test_results_splits, f, protocol=pickle.HIGHEST_PROTOCOL)