import os
import json
import argparse
import pickle
from tqdm import tqdm

import numpy as np
import torch
from torchvision import transforms
import h5py

import pdb 

from wsi_model import *
from read_data import *
from resnet import resnet50


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Getting features')

    parser.add_argument('--ref_file', type=str, required=True, help='Reference file')
    parser.add_argument('--patch_data_path', type=str, required=True, help='Directory where the patch is saved')
    parser.add_argument('--feature_path', type=str, default="/oak/stanford/groups/ogevaert/data/Gen-Pred/features/", help='Output directory to save features')
    parser.add_argument('--max_patch_number', type=int, default=4000, help='Max number of patches to use per slide')
    parser.add_argument('--seed', type=int, default=99,
            help='Seed for random generation')
    parser.add_argument("--tcga_projects", help="the tcga_projects we want to use",
                        default=None, type=str, nargs='*')
    parser.add_argument('--start', type=int, default=None,
                        help='Index for start slide')
    parser.add_argument('--end', type=int, default=None,
                        help='Index for ending slide')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    print(10*'-')
    print('Args for this experiment \n')
    print(args)
    print(10*'-')

    random.seed(args.seed)

    path_csv = args.ref_file
    patch_data_path = args.patch_data_path

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transforms_val = torch.nn.Sequential(
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    resnet50 = resnet50(pretrained=True).to(device)

    resnet50.eval()
    print('Loading dataset...')

    df = pd.read_csv(path_csv)
    # there could be duplicated WSIs mapped to different RNA files and we only need features for each WSI
    df = df.drop_duplicates(["wsi_file_name"]) 

    # Filter tcga projects
    if args.tcga_projects:
        df = df[df['tcga_project'].isin(args.tcga_projects)]
    
    # indexing based on values for parallelization
    if args.start is not None and args.end is not None:
        df = df.iloc[args.start:args.end]
    elif args.start is not None:
        df = df.iloc[args.start:]
    elif args.end is not None:
        df = df.iloc[:args.end]

    print(f'Number of slides = {df.shape[0]}')

    for i, row in tqdm(df.iterrows()):
        WSI = row['wsi_file_name']
        WSI_slide = WSI.split('.')[0]
        project = row['tcga_project'] 
        WSI = WSI.replace('.svs', '') # in the ref file of prad there is a .svs that should not be there

        if not os.path.exists(os.path.join(patch_data_path, WSI_slide)):
            print('Not exist {}'.format(os.path.join(patch_data_path, WSI_slide)))
            continue

        path = os.path.join(patch_data_path, WSI_slide, WSI_slide + '.hdf5')
        path_h5 = os.path.join(args.feature_path, project, WSI)

        if not os.path.exists(path_h5):
            os.makedirs(path_h5)

        if os.path.exists(os.path.join(path_h5, "complete_resnet.txt")):
            print(f'{WSI}: Resnet features already obtained')
            continue

        try:
            with h5py.File(path, 'r') as f_read:
                keys = list(f_read.keys())
                if len(keys) > args.max_patch_number:
                    keys = random.sample(keys, args.max_patch_number)

                features_tiles = []
                for key in tqdm(keys):
                    image = f_read[key][:]
                    image = torch.from_numpy(image).permute(2,0,1)
                    image = transforms_val(image).to(device)
                    with torch.no_grad():
                        features = resnet50.forward_extract(image[None,:])
                        features_tiles.append(features[0].detach().cpu().numpy())

            features_tiles = np.asarray(features_tiles)
            n_tiles = len(features_tiles)

            f_write = h5py.File(os.path.join(path_h5, WSI+'.h5'), "w")
            dset = f_write.create_dataset("resnet_features", data=features_tiles)
            f_write.close()

            with open(os.path.join(path_h5, "complete_resnet.txt"), 'w') as f_sum:
                f_sum.write(f"Total n patch = {n_tiles}")

        except Exception as e:
            print(e)
            print(WSI)
            continue
            