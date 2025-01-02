# Make predictions using the trained models

import os
import argparse
from tqdm import tqdm
import pickle
# from he2rna import HE2RNA, he2rna_predict
# from vit import ViT
import h5py
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn import preprocessing
from accelerate import Accelerator
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import torch.nn as nn

from read_data import SuperTileRNADataset
from utils import patient_kfold, filter_no_features, custom_collate_fn
from src.tformer_lin import ViS

def custom_collate_fn(batch):
    """Remove bad entries from the dataloader
    Args:
        batch (torch.Tensor): batch of tensors from the dataaset
    Returns:
        collate: Default collage for the dataloader
    """
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Getting features')
    parser.add_argument('--ref_file', type=str, required=True, help='Reference file')
    parser.add_argument('--feature_path', type=str, default="features/", help='path to resnet/uni and clustered features')
    parser.add_argument('--feature_use', type=str, default="cluster_mean_features", help='which feature to use for training the model')
     parser.add_argument('--checkpoint', type=str, default=None, help='Path to the trained models')
    parser.add_argument('--folds', type=int, default=5, help='Folds for pre-trained model')
    parser.add_argument('--seed', type=int, default=99, help='Seed for random generation')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--depth', type=int, default=6, help='transformer depth')
    parser.add_argument('--num-heads', type=int, default=16, help='number of attention heads')
    parser.add_argument('--num_clusters', type=int, default=100, help='Number of clusters for the kmeans')
    parser.add_argument('--tcga_projects', default=None, type=str, nargs='*', help="the tcga_projects we want to use")
    parser.add_argument('--save_dir', type=str, default="/oak/stanford/groups/ogevaert/data/Gen-Pred/vit_exp", help='save_path')
    parser.add_argument('--exp_name', type=str, default="exp", help='Experiment name')

    ############################################## variables ##############################################
    args = parser.parse_args()

    ############################################## seeds ##############################################
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ############################################## run args ##############################################

    save_dir = os.path.join(args.save_dir, args.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    run = None
    #run = wandb.init(project="visgene", entity='mpizuric', config=args, name=args.exp_name) 
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)

    ############################################## data prep ##############################################
    path_csv = args.ref_file
    df = pd.read_csv(path_csv)

    # filter out WSIs for which we don't have features
    df = filter_no_features(df, feature_path = args.feature_path, feature_name = args.feature_use)

    if 'tcga_project' in df.columns and args.tcga_projects:
        df = df[df['tcga_project'].isin(args.tcga_projects)].reset_index(drop=True)
    
    genes = [c[4:] for c in df.columns if "rna_" in c]
    test_dataset = SuperTileRNADataset(df, args.feature_path, args.feature_use)

    test_dataloader = DataLoader(test_dataset, 
                num_workers=0, pin_memory=True, 
                shuffle=False, batch_size=args.batch_size,
                collate_fn=custom_collate_fn)

    feature_dim = test_dataset.feature_dim
    print(f'Feature dimension: {feature_dim}')

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

            model = ViS(num_outputs=test_dataset.num_genes, 
                        input_dim=feature_dim, 
                        depth=args.depth, nheads=args.num_heads,  
                        dimensions_f=64, dimensions_c=64, dimensions_s=64, device=device)
            model.load_state_dict(torch.load(pretrained_model, map_location=device))
            model.to(device)
            preds, wsis, projs = predict(model, test_dataloader, run=run)
            random_model = ViS(num_outputs=test_dataset.num_genes, 
                                input_dim=feature_dim, 
                                depth=args.depth, nheads=args.num_heads,  
                                dimensions_f=64, dimensions_c=64, dimensions_s=64, device=device)
            random_model.to(device)
            random_preds, _, _ = predict(random_model, test_dataloader, run=run)

        res_preds.append(preds)
        res_random.append(random_preds)

    # calculate average across folds
    avg_preds = np.mean(res_preds, axis = 0)
    avg_random = np.mean(res_random, axis = 0)

    df_pred = pd.DataFrame(avg_preds, index = wsis, columns = genes)
    df_random = pd.DataFrame(avg_random, index = wsis, columns = genes)

    test_results = {'pred': df_pred, 'random': df_random}

    with open(os.path.join(save_dir, 'test_results.pkl'), 'wb') as f:
        pickle.dump(test_results, f, protocol=pickle.HIGHEST_PROTOCOL)