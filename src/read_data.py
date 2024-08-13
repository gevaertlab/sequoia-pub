import os
import pickle
import random

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch
from tqdm import tqdm
import lmdb
import lz4framed
import cv2
import h5py
import pdb


class SuperTileRNADataset(Dataset):
    def __init__(self, csv_path: str, features_path, quick=None):
        self.csv_path = csv_path
        self.quick = quick
        self.features_path = features_path
        if type(csv_path) == str:
            self.data = pd.read_csv(csv_path)
        else:
            self.data = csv_path

        # find the number of genes
        row = self.data.iloc[0]
        rna_data = row[[x for x in row.keys() if 'rna_' in x]].values.astype(np.float32)
        self.num_genes = len(rna_data)

        # find the feature dimension, assume all images in the reference file have the same dimension
        path = os.path.join(self.features_path, row['tcga_project'], 
                            row['wsi_file_name'], row['wsi_file_name']+'.h5')
        f = h5py.File(path, 'r')
        features = f[self.feature_use][:]
        self.feature_dim = features.shape[1]
        f.close()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = os.path.join(self.features_path, row['tcga_project'],
                            row['wsi_file_name'], row['wsi_file_name']+'.h5')
        rna_data = row[[x for x in row.keys() if 'rna_' in x]].values.astype(np.float32)
        rna_data = torch.tensor(rna_data, dtype=torch.float32)
        try:
            if 'GTEX' not in path:
                path = path.replace('.svs','')
            f = h5py.File(path, 'r')
            features = f['cluster_features'][:]
            f.close()
            features = torch.tensor(features, dtype=torch.float32)
        except Exception as e:
            print(e)
            print(path)
            features = None

        return features, rna_data, row['wsi_file_name'], row['tcga_project']


def split_train_test_ids(df, test_ids_path):
    file_parse = np.loadtxt(test_ids_path, dtype=str)
    test_ids = [x.split('"')[1] for x in file_parse]
    wsi_file_names = df['wsi_file_name'].values

    test_wsi = []
    for test_id in test_ids:
        if test_id in wsi_file_names:
            test_wsi.append(test_id)

    test_df = df.loc[df['wsi_file_name'].isin(test_wsi)]
    train_df = df.loc[~df['wsi_file_name'].isin(test_wsi)]

    return train_df, test_df