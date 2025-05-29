"""
HE2RNA: definition of the algorithm to generate a model for gene expression prediction
Copyright (C) 2020  Owkin Inc.
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from tkinter.messagebox import NO
import numpy as np
import pandas as pd
import torch
import time
import os
from torch import nn
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter
from tqdm import tqdm
import datetime
import wandb
import argparse
import json
from sklearn.model_selection import train_test_split
#from accelerate import Accelerator
from einops import rearrange
import pickle
import h5py

from huggingface_hub import PyTorchModelHubMixin

from src.read_data import SuperTileRNADataset
from src.utils import patient_split, patient_kfold, custom_collate_fn, filter_no_features


class HE2RNA(nn.Module, PyTorchModelHubMixin):
    """Model that generates one score per tile and per predicted gene.
    Args
        output_dim (int): Output dimension, must match the number of genes to
            predict.
        layers (list): List of the layers' dimensions
        nonlin (torch.nn.modules.activation)
        ks (list): list of numbers of highest-scored tiles to keep in each
            channel.
        dropout (float)
        device (str): 'cpu' or 'cuda'
        mode (str): 'binary' or 'regression'
    """
    def __init__(self, input_dim, output_dim,
                 layers=[1], nonlin=nn.ReLU(), ks=[10],
                 dropout=0.5, device='cpu',
                 bias_init=None, **kwargs):
        super(HE2RNA, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = [input_dim] + layers + [output_dim]
        self.layers = []
        for i in range(len(layers) - 1):
            layer = nn.Conv1d(in_channels=layers[i],
                              out_channels=layers[i+1],
                              kernel_size=1,
                              stride=1,
                              bias=True)
            setattr(self, 'conv' + str(i), layer)
            self.layers.append(layer)
        if bias_init is not None:
            self.layers[-1].bias = bias_init
        self.ks = np.array(ks)

        self.nonlin = nonlin
        self.do = nn.Dropout(dropout)
        self.device = device
        self.to(self.device)

    def forward(self, x):
        if self.training:
            k = int(np.random.choice(self.ks))
            return self.forward_fixed_k(x, k)
        else:
            pred = 0
            for k in self.ks:
                pred += self.forward_fixed_k(x, int(k)) / len(self.ks)
            return pred

    def forward_fixed_k(self, x, k):
        mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = (mask > 0).float()
        x = self.conv(x) * mask
        t, _ = torch.topk(x, k, dim=2, largest=True, sorted=True)
        x = torch.sum(t * mask[:, :, :k], dim=2) / torch.sum(mask[:, :, :k], dim=2)
        return x

    def conv(self, x):
        x = x[:, x.shape[1] - self.input_dim:]
        for i in range(len(self.layers) - 1):
            x = self.do(self.nonlin(self.layers[i](x)))
        x = self.layers[-1](x)
        return x

def training_epoch(model, dataloader, optimizer):
    """Train model for one epoch.
    """
    model.train()
    loss_fn = nn.MSELoss()
    train_loss = []
    for x, y, _, _ in tqdm(dataloader):
        x = x.float().to(model.device)
        # rearranging dimenions (b, c, f) to (b, c*f)
        x = rearrange(x, 'b c f -> b f c')
        y = y.float().to(model.device)
        pred = model(x)
        loss = loss_fn(pred, y)
        train_loss += [loss.detach().cpu().numpy()]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = np.mean(train_loss)
    return train_loss

'''
def compute_correlations(labels, preds, projects):
    metrics = []
    for project in np.unique(projects):
        for i in range(labels.shape[1]):
            y_true = labels[projects == project, i]
            if len(np.unique(y_true)) > 1:
                y_prob = preds[projects == project, i]
                metrics.append(np.corrcoef(y_true, y_prob)[0, 1])
    metrics = np.asarray(metrics)
    return np.mean(metrics)
'''
def compute_correlations(labels, preds):
    metrics = []
    for i in range(labels.shape[1]):
        y_true = labels[:, i]
        if len(np.unique(y_true)) > 1:
            y_prob = preds[:, i]
            metrics.append(np.corrcoef(y_true, y_prob)[0, 1])
    metrics = np.asarray(metrics)
    metrics = metrics[~np.isnan(metrics)]
    return np.mean(metrics)

def evaluate(model, dataloader):
    """Evaluate the model on the validation set and return loss and metrics.
    """
    model.eval()
    loss_fn = nn.MSELoss()
    valid_loss = []
    preds = []
    labels = []
    for x, y, _, _ in dataloader:
        # rearranging dimenions (b, c, f) to (b, c*f)
        x = x.float().to(model.device)
        x = rearrange(x, 'b c f -> b f c')
        pred = model(x)
        labels += [y]
        loss = loss_fn(pred, y.float().to(model.device))
        valid_loss += [loss.detach().cpu().numpy()]
        pred = nn.ReLU()(pred)
        preds += [pred.detach().cpu().numpy()]
    valid_loss = np.mean(valid_loss)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    metrics = compute_correlations(labels, preds)
    return valid_loss, metrics

def he2rna_predict(model, dataloader):
    """Perform prediction on the test set.
    """
    model.eval()
    preds = []
    wsis = []
    projs = []
    labels = []
    for x, y,  wsi_file_name, tcga_project in dataloader:
        x = x.float().to(model.device)
        wsis.append(wsi_file_name)
        projs.append(tcga_project)
        # rearranging dimenions (b, c, f) to (b, c*f)
        x = rearrange(x, 'b c f -> b f c')
        pred = model(x)
        pred = nn.ReLU()(pred)
        preds += [pred.detach().cpu().numpy()]
        labels += [y]
    preds = np.concatenate((preds), axis=0)
    wsis = np.concatenate((wsis), axis=0)
    projs = np.concatenate((projs), axis=0)
    labels = np.concatenate((labels), axis=0)
    return preds, labels, wsis, projs

# def predict(model, dataloader):
#     """Perform prediction on the test set.
#     """
#     model.eval()
#     labels = []
#     preds = []
#     for x, y, _ , _ in dataloader:
#         # rearranging dimenions (b, c, f) to (b, c*f)
#         x = x.float().to(model.device)
#         x = rearrange(x, 'b c f -> b f c')
#         pred = model(x)
#         labels += [y]
#         pred = nn.ReLU()(pred)
#         preds += [pred.detach().cpu().numpy()]
#     preds = np.concatenate(preds)
#     labels = np.concatenate(labels)
#     return preds, labels

def fit(model,
        lr,
        train_loader,
        valid_loader,
        test_loader,
        params={},
        fold=None,
        optimizer=None,
        path=None):
    """Fit the model and make prediction on evaluation set.
    Args:
        model (nn.Module)
        train_set (torch.utils.data.Dataset)
        valid_set (torch.utils.data.Dataset)
        params (dict): Dictionary for specifying training parameters.
            keys are 'max_epochs' (int, default=200), 'patience' (int,
            default=20) and 'batch_size' (int, default=16).
        optimizer (torch.optim.Optimizer): Optimizer for training the model
        test_set (None or torch.utils.data.Dataset): If None, return
            predictions on the validation set.
        path (str): Path to the folder where th model will be saved.
        logdir (str): Path for TensoboardX.
    """

    if path is not None and not os.path.exists(path):
        os.mkdir(path)

    default_params = {
        'max_epochs': 200,
        'patience': 100}
    default_params.update(params)
    patience = default_params['patience']
    max_epochs = default_params['max_epochs']

    if optimizer is None:
        optimizer = torch.optim.Adam(list(model.parameters()), lr=lr, weight_decay=0.)

    metrics = 'correlations'
    epoch_since_best = 0
    start_time = time.time()

    if valid_loader != None:
        valid_loss, best = evaluate(model, valid_loader)
        print('{}: {:.3f}'.format(metrics, best))

        if np.isnan(best):
            best = 0
    else:
        best = 0

    name = 'model'
    if fold != None:
        name = name + '_' + str(fold)

    try:
        for e in range(max_epochs):

            epoch_since_best += 1

            train_loss = training_epoch(model, train_loader, optimizer)
            dic_loss = {'train_loss': train_loss}

            print('Epoch {}/{} - {:.2f}s'.format(e + 1, max_epochs, time.time() - start_time))
            start_time = time.time()

            if valid_loader != None:
                valid_loss, scores = evaluate( model, valid_loader)
                dic_loss['valid_loss'] = valid_loss
                score = np.mean(scores)

                if args.log:
                    wandb.log({'epoch': e, 'score '+str(fold): score})
                    wandb.log({'epoch': e, 'valid loss fold '+str(fold): valid_loss})
                    wandb.log({'epoch': e, 'train loss fold '+str(fold): train_loss})

                print('loss: {:.4f}, val loss: {:.4f}'.format(train_loss, valid_loss))
                print('{}: {:.3f}'.format(metrics, score))

                criterion = (score > best)

                if criterion:
                    epoch_since_best = 0
                    best = score
                    if path is not None:
                        torch.save(model, os.path.join(path, name + '.pt'))

                if epoch_since_best == patience:
                    print('Early stopping at epoch {}'.format(e + 1))
                    break

    except KeyboardInterrupt:
        pass

    if path is not None and os.path.exists(os.path.join(path, name + '.pt')):
        model = torch.load(os.path.join(path, name + '.pt'))

    elif path is not None:
        torch.save(model, os.path.join(path, name + '.pt'))

    if (test_loader != None):
        preds_test, labels_test, wsis, projs = he2rna_predict(model, test_loader)
        return preds_test, labels_test, wsis, projs

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Getting features')
    parser.add_argument('--path_csv', type=str, help='path to csv file with gene expression info')
    parser.add_argument('--feature_path', type=str, default="features/", help='path to resnet/uni and clustered features')
    parser.add_argument('--checkpoint', type=str, help='pretrained model path')
    parser.add_argument('--change_num_genes', help="whether finetuning from a model trained on different number of genes", action="store_true")
    parser.add_argument('--num_genes', type=int, help='number of genes in output of pretrained model')
    parser.add_argument('--seed', type=int, default=99, help='Seed for random generation')
    parser.add_argument('--log', type=int, default=1, help='Whether to do the log with wandb')
    parser.add_argument('--k', type=int, default=5, help='Number of splits')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--num_workers', type=int, default=0, help='num workers dataloader')
    parser.add_argument('--tcga_projects', help="the tcga_projects we want to use", default=None, type=str, nargs='*')
    parser.add_argument('--exp_name', type=str, default="exp", help='Experiment name')
    parser.add_argument('--subfolder', type=str, default="", help='subfolder where result will be saved')
    parser.add_argument('--destfolder', type=str, default="", help='destination folder')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_dir = os.path.join(args.destfolder, args.subfolder, args.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    experiment_name = args.exp_name

    if args.log:
        run = wandb.init(project="sequoia", entity='entity_name', config=args, name=experiment_name)

    path_csv = args.path_csv

    df = pd.read_csv(path_csv)
    if args.tcga_projects:
        df = df[df['tcga_project'].isin(args.tcga_projects)]

    ############################################## data prep ##############################################
    # filter out WSIs for which we don't have features
    df = filter_no_features(df, args.feature_path, 'cluster_features')

    ############################################## model training ##############################################v

    train_idxs, val_idxs, test_idxs = patient_kfold(df, n_splits=args.k)
    test_results_splits = {}
    i = 0
    for train_idx, val_idx, test_idx in zip(train_idxs, val_idxs, test_idxs):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        test_df = df.iloc[test_idx]

        train_dataset = SuperTileRNADataset(train_df, args.feature_path)
        val_dataset = SuperTileRNADataset(val_df, args.feature_path)
        test_dataset = SuperTileRNADataset(test_df, args.feature_path)

        # SET num_workers TO 0 WHEN WORKING WITH hdf5 FILES
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate_fn)

        valid_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)

        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not args.change_num_genes:
            model = HE2RNA(input_dim=2048, layers=[256,256],
                    ks=[1,2,5,10,20,50,100],
                    output_dim=train_dataset.num_genes, device=device)
        else:
            model = HE2RNA(input_dim=2048, layers=[256,256],
                    ks=[1,2,5,10,20,50,100],
                    output_dim=args.num_genes, device=device)

        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint).state_dict())

        if args.change_num_genes: # change num genes: num genes was different for pretraining on gtex
            model.conv2 = nn.Conv1d(in_channels=model.conv1.out_channels,
                                        out_channels=train_dataset.num_genes,
                                        kernel_size=1,
                                        stride=1,
                                        bias=True).to(device)
            model.layers[-1] = model.conv2

        preds_random, labels_random, wsis_rand, projs_rand = he2rna_predict(model, test_loader)

        preds, labels, wsis, projs = fit(model=model,
                                            lr=args.lr,
                                            train_loader=train_loader,
                                            valid_loader=valid_loader,
                                            test_loader=test_loader,
                                            params={},
                                            fold=i,
                                            optimizer=None,
                                            path=save_dir)

        test_results = {
            'real': labels,
            'preds': preds,
            'random': preds_random,
            'wsi_file_name': wsis,
            'tcga_project': projs
        }

        test_results_splits[f'split_{i}'] = test_results
        i += 1

    test_results_splits['genes'] = [x[4:] for x in df.columns if 'rna_' in x]
    with open(os.path.join(save_dir, 'test_results.pkl'), 'wb') as f:
        pickle.dump(test_results_splits, f, protocol=pickle.HIGHEST_PROTOCOL)


