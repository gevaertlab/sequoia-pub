import os
import json
import argparse
import pdb
from tqdm import tqdm

import numpy as np
import h5py
from sklearn.cluster import KMeans

from read_data import *
from utils import exists

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Getting features')
    parser.add_argument('--ref_file', type=str, required=True, help='Path with reference csv file')
    parser.add_argument('--patch_data_path', type=str, required=True, help='Directory where the patch is saved')
    parser.add_argument('--feature_path', type=str, default="/oak/stanford/groups/ogevaert/data/Gen-Pred/features/", help='Output directory to save features')
    parser.add_argument('--num_clusters', type=int, default=100,
                        help='Number of clusters for the kmeans')
    parser.add_argument("--tcga_projects", help="the tcga_projects we want to use",
                        default=None, type=str, nargs='*')
    parser.add_argument('--start', type=int, default=0,
                        help='Index for start slide')
    parser.add_argument('--end', type=int, default=None,
                        help='Index for ending slide')
    parser.add_argument("--gtex", help="using gtex data",
                    action="store_true")
    parser.add_argument('--gtex_tissue', type=str, default=None,
                        help='GTex tissue being used')
    parser.add_argument('--seed', type=int, default=99,
        help='Seed for random generation')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(10*'-')
    print('Args for this experiment \n')
    print(args)
    print(10*'-')

    path_csv = args.ref_file

    df = pd.read_csv(path_csv)
    df = df.drop_duplicates(['wsi_file_name'])

    # Filter tcga projects
    if args.tcga_projects:
        df = df[df['tcga_project'].isin(args.tcga_projects)]

    print(f'Total number of slides = {df.shape[0]}')    

    # indexing based on values for parallelization
    if exists(args.start) and exists(args.end):
        df = df.iloc[args.start:args.end]
    elif exists(args.start):
        df = df.iloc[args.start:]
    elif args.end:
        df = df.iloc[:args.end]

    print(f'New number of slides = {df.shape[0]}')

    # ##### get matching IDs for slides where we have both genes and patches
    # patch_data_path = args.patch_data_path
    # gene_ids = [i if ('.svs' in i) else i+'.svs' for i in df['wsi_file_name'].values]
    # patch_ids = os.listdir(patch_data_path)
    # patch_ids = [i if ('.svs' in i) else i+'.svs' for i in patch_ids]

    # matching = [i for i in gene_ids if i in patch_ids]
    # not_matching = [i for i in gene_ids if i not in patch_ids]
    # print('Amount matching: '+str(len(matching))+', amount not: '+str(len(not_matching)))
    # new_matching = []
    # if args.gtex:
    #     # Globus filename ID has 25 instead of 26 sometimes at the end
    #     changed_ids = ['-'.join(i.split('-')[:2])+'-'+i.split('-')[-1][:2]+i.split('-')[-1][2:4].replace('26','25')+'.svs' for i in not_matching]
    #     new_matching = [i for i in changed_ids if i in patch_ids]
    #     print('Amount new matching: '+str(len(new_matching)))
    
    # all_wsis_matching = matching + new_matching
    # print('Total amount matching: '+str(len(all_wsis_matching)))
   # for WSI in tqdm(all_wsis_matching):

    for i, row in tqdm(df.iterrows()):
        WSI = row['wsi_file_name']
        if args.gtex:
            project = args.gtex_tissue
        else:
            project = df.iloc[0]['tcga_project'] 
            WSI = WSI.replace('.svs', '')

        path = os.path.join(args.feature_path, project, WSI)
        try:
            f = h5py.File(os.path.join(path,WSI+'.h5'), "r+")
        except:
            print(f'Cannot open file {path}')
            continue
        try:
            features = f['resnet_features']
        except:
            print(f'No resnet features for {path}')
            f.close()
            continue
        
        if features.shape[0] < args.num_clusters:
            print(f'{WSI} less number of patches than clusters')
            f.close()
            continue
        
        if 'cluster_features' in f.keys():
            print(f'{WSI}: Cluster feature already available')
            f.close()
            continue
        
        kmeans = KMeans(n_clusters=args.num_clusters, random_state=0).fit(features)
        clusters = kmeans.labels_

        mean_features = []
        for pos in tqdm(range(args.num_clusters)):
            indexes = np.where(clusters == pos)
            features_aux = features[indexes]
            mean_features.append(np.mean(features_aux, axis=0))
        
        mean_features = np.asarray(mean_features)
        
        try:
            dset = f.create_dataset("cluster_features", data=mean_features)
            f.close()
        except Exception as e:
            print(f"{WSI}: Error in creating cluster_feauture")
            print(e)
            f.close()
    print('Done!')