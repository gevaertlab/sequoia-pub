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

from typing import Any

class PatchBagRNADataset(Dataset):
    def __init__(self, patch_data_path: str, csv_path: str, img_size:int , 
                    transforms=None, max_patch_per_wsi=400, bag_size=20,
                    quick=None, label_encoder=None, type='classification',
                    ordinal=False):
        self.patch_data_path = patch_data_path
        self.csv_path = csv_path
        self.img_size = img_size
        self.bag_size = bag_size
        self.transforms = transforms
        self.max_patches_total = max_patch_per_wsi
        self.quick = quick
        self.le = label_encoder
        self.ordinal = ordinal
        self.type = type
        self.data = {}
        self.index = []
        self._preprocess()

    def _preprocess(self):
        if type(self.csv_path) == str:
            csv_file = pd.read_csv(self.csv_path)
        else:
            csv_file = self.csv_path
        
        if self.quick:
            csv_file = csv_file.sample(150)
        
        for i, row in tqdm(csv_file.iterrows()):
            WSI = row['wsi_file_name']
            label = np.asarray(row['Labels'])
            if label == "'--": continue
            rna_data = row[[x for x in row.keys() if 'rna_' in x]].values.astype(np.float32)
            rna_data = torch.tensor(rna_data, dtype=torch.float32)
            if self.le is not None:
                label = self.le.transform(label.reshape(-1,1))
                if self.type == 'regression':
                    label = label.astype(np.float32)
            else:
                if self.ordinal:
                    label = label.astype(np.int64)
                else:
                    label = label.astype(np.float32)

            project = row['tcga_project'] 
            if not os.path.exists(os.path.join('../'+project+self.patch_data_path, WSI)):
                print('Not exist {}'.format(os.path.join('../'+project+self.patch_data_path, WSI)))
                continue
            
            #try:
            path = os.path.join('../'+project+self.patch_data_path, WSI, WSI)
            try:
                lmdb_connection = lmdb.open(path,
                                            subdir=False, readonly=True, 
                                            lock=False, readahead=False, meminit=False)
            except:
                path = path + '.db'
                try:
                    lmdb_connection = lmdb.open(path,
                                                subdir=False, readonly=True, 
                                                lock=False, readahead=False, meminit=False)
                except:
                    continue
            try:
                with lmdb_connection.begin(write=False) as lmdb_txn:
                    n_patches = lmdb_txn.stat()['entries'] - 1
                    keys = pickle.loads(lz4framed.decompress(lmdb_txn.get(b'__keys__')))
            except Exception as e:
                print(e)
                continue

            #except:
            #    print('Error with loc file {}'.format(os.path.join('../'+project+self.patch_data_path, WSI)))
            #    continue
            n_selected = min(n_patches, self.max_patches_total)
            n_patches= list(range(n_patches))
            images = random.sample(n_patches, n_selected)
            new_row = dict()
            new_row['WSI'] = WSI
            new_row['rna_data'] = rna_data
            new_row['label'] = label
            self.data[WSI] = {w.lower(): new_row[w] for w in new_row.keys()}
            self.data[WSI].update({'WSI': WSI, 'images': images, 'n_images': len(images),
                                   'lmdb_path': path, 'keys': keys})
            for k in range(len(images) // self.bag_size):
                self.index.append((WSI, self.bag_size * k))

    def decompress_and_deserialize(self, lmdb_value: Any):
        try:
            img_name, img_arr, img_shape = pickle.loads(lz4framed.decompress(lmdb_value))
        except:
            return None
        image = np.frombuffer(img_arr, dtype=np.uint8).reshape(img_shape)
        image = np.copy(image)
        return torch.from_numpy(image).permute(2,0,1)

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        (WSI, i) = self.index[idx]
        imgs = []
        row = self.data[WSI].copy()
        lmdb_connection = lmdb.open(row['lmdb_path'],
                                        subdir=False, readonly=True, 
                                        lock=False, readahead=False, meminit=False)
        with lmdb_connection.begin(write=False) as txn:
            for patch in row['images'][i:i + self.bag_size]:
                lmdb_value = txn.get(row['keys'][patch])
                img = self.decompress_and_deserialize(lmdb_value)
                imgs.append(img)
        img = torch.stack(imgs,dim=0)
        return img, row['rna_data'], row['label']

class PatchDataset(Dataset):
    def __init__(self, patch_data_path, csv_path, img_size, transforms=None,
            max_patches_total=300, quick=False, le=None):
        self.patch_data_path = patch_data_path
        self.csv_path = csv_path
        self.img_size = img_size
        self.transforms = transforms
        self.max_patches_total = max_patches_total
        self.quick = quick
        self.keys = []
        self.images = []
        self.filenames = []
        self.labels = []
        self.lmdbs_path = []
        self.le = le
        self._preprocess()

    def _preprocess(self):
        if type(self.csv_path) == str:
            csv_file = pd.read_csv(self.csv_path)
            csv_file['patch_data_path'] = [self.patch_data_path] * csv_file.shape[0]
            csv_file['labels'] = [0] * csv_file.shape[0]
        else:
            csv_file = self.csv_path
        
        if self.quick:
            csv_file = csv_file.sample(10)
        
        for i, row in tqdm(csv_file.iterrows()):
            row = row.to_dict()
            WSI = row['wsi_file_name']
            '''
            label = np.asarray(row['labels'])
            if self.le is not None:
                label = self.le.transform(label.reshape(-1,1))
            label = torch.tensor(label, dtype=torch.float32)
            '''
            #label = label.flatten()
            project = row['tcga_project'] 
            if not os.path.exists(os.path.join('../../Roche-TCGA/'+project+self.patch_data_path, WSI)):
                print('Not exist {}'.format(os.path.join('../'+project+self.patch_data_path, WSI)))
                continue
            path = os.path.join('../../Roche-TCGA/'+project+self.patch_data_path, WSI, WSI)
            try:
                lmdb_connection = lmdb.open(path,
                                            subdir=False, readonly=True, 
                                            lock=False, readahead=False, meminit=False)
            except:
                path = path + '.db'
                try:
                    lmdb_connection = lmdb.open(path,
                                                subdir=False, readonly=True, 
                                                lock=False, readahead=False, meminit=False)
                except:
                    continue
            try:
                with lmdb_connection.begin(write=False) as lmdb_txn:
                    n_patches = lmdb_txn.stat()['entries'] - 1
                    keys = pickle.loads(lz4framed.decompress(lmdb_txn.get(b'__keys__')))
                #n_patches = sum(1 for _ in open(os.path.join(data_path, WSI, 'loc.txt'))) - 2
                n_selected = min(n_patches, self.max_patches_total)
                n_patches= list(range(n_patches))
                n_patches_index = random.sample(n_patches, n_selected)
            
            except Exception as e:
                print(e)
                continue   

            for i in n_patches_index:
                #self.images.append(os.path.join(data_path, WSI, WSI + '_patch_{}.png'.format(i)))
                self.images.append(i)
                self.filenames.append(WSI)
                self.labels.append(project)
                self.lmdbs_path.append(path)
                self.keys.append(keys[i])

    def decompress_and_deserialize(self, lmdb_value: Any):
        try:
            img_name, img_arr, img_shape = pickle.loads(lz4framed.decompress(lmdb_value))
        except Exception as e:
            print(e)
            return None
        image = np.frombuffer(img_arr, dtype=np.uint8).reshape(img_shape)
        image = np.copy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image).permute(2,0,1)
     
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        lmdb_connection = lmdb.open(self.lmdbs_path[idx],
                                        subdir=False, readonly=True, 
                                        lock=False, readahead=False, meminit=False)
        
        with lmdb_connection.begin(write=False) as txn:
            lmdb_value = txn.get(self.keys[idx])

        image = self.decompress_and_deserialize(lmdb_value)

        if image == None:
            print(self.lmdbs_path[idx])
            return None

        out = {
            'image': self.transforms(image),
            'label': self.labels[idx],
            'filename': self.filenames[idx]
        }
        return out

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


class SuperTilePathwayDataset(Dataset):
    def __init__(self, df, features_path:str, pathway_code:str):
        '''
        pathway_code needs to be any one of wnt,pi3k,p53,notch,rtk,nrf2,tgfb,myc,cellcycle,hippo
        '''

        self.features_path = features_path
        self.data = df
        self.pathway_code = pathway_code
        self.targets = self.data[pathway_code].values
   
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = os.path.join(self.features_path, row['tcga_project'], 
                            row['wsi_file_name'], row['wsi_file_name']+'.h5')
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
           
        return features, row[self.pathway_code], row['wsi_file_name'], row['tcga_project']
    

class SuperTileCellStatesDataset(Dataset):
    def __init__(self, df, features_path):
        self.features_path = features_path
        self.data = df

        # find the number of cell states 
        cell_cols = [col for col in self.data.columns if ('S0' in col) and ('cells' in col)]
        self.num_cell_states = len(cell_cols)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = os.path.join(self.features_path, row['tcga_project'], 
                            row['wsi_file_name'], row['wsi_file_name']+'.h5')
        cellstates = row[[col for col in self.data.columns if ('S0' in col) and ('cells' in col)]].values.astype(np.float32)
        cellstates = torch.tensor(cellstates, dtype=torch.float32)
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
           
        return features, cellstates, row['wsi_file_name'], row['tcga_project']