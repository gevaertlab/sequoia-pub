import os
import argparse

import numpy as np
import pandas as pd
from einops import rearrange

from scipy.ndimage.morphology import binary_dilation
import openslide

import torch
from torchvision import transforms

from tqdm import tqdm

import sys
sys.path.append('./src/')
sys.path.append('./src/results_analysis/')

from results_analysis.analyze_significance import read_pickle
from vit_new import ViT
from resnet import resnet50

BACKGROUND_THRESHOLD = .5

def sliding_window_method(df, patch_size_resized, 
                            resnet50, model, inds_gene_of_interest, stride, device='cpu'):

    max_x = max(df['xcoord_tf'])
    max_y = max(df['ycoord_tf'])

    preds = {} # {key:value} where key is a gene index and value is a new dict that contains the predictions per tile for that gene
    for ind_gene in inds_gene_of_interest:
        preds[ind_gene] = {}

    for x in tqdm(range(0, max_x, stride)):
        for y in range(0, max_y, stride):
            
            window = df[((df['xcoord_tf']>=x) & (df['xcoord_tf']<(x+10))) &
                        ((df['ycoord_tf']>=y) & (df['ycoord_tf']<(y+10)))]

            if window.shape[0] > ((10*10)/2):
                # get the patches
                features_all = []
                for ind in window.index:
                    col = df.iloc[ind]['xcoord']
                    row = df.iloc[ind]['ycoord']
                    patch = slide.read_region((col, row), 0, (patch_size_resized, patch_size_resized)).convert('RGB')
                    patch_tf = transforms_(patch).unsqueeze(0).to(device)
                    with torch.no_grad():
                        features = resnet50.forward_extract(patch_tf)
                        features_all.append(features)

                # if window contains less than 10x10 tiles, pad with 0
                features_all = torch.cat(features_all)
                if features_all.shape[0] < 100:
                    padding = torch.cat([torch.zeros(1,2048) for _ in range(100-features_all.shape[0])]).to(device)
                    features_all = torch.cat([features_all, padding])

                # get ViT predictions
                # model_predictions = predict(model, random_sample_patches)
                with torch.no_grad():
                    model_predictions = model(features_all)
                    
                predictions = model_predictions.detach().cpu().numpy()[0]

                # add predictions to dict (same for all tiles in window)
                for ind_gene in inds_gene_of_interest:
                    for _, key in enumerate(window.index):
                        if stride == 10:
                            preds[ind_gene][key] = predictions[ind_gene] 
                        else:
                            if key not in preds[ind_gene].keys():
                                preds[ind_gene][key] = [predictions[ind_gene]]
                            else:
                                preds[ind_gene][key].append(predictions[ind_gene])

    if stride < 10:
        for ind_gene in inds_gene_of_interest:
            for key in preds[ind_gene].keys():
                preds[ind_gene][key] = np.mean(preds[ind_gene][key])

    return preds

if __name__=='__main__':

    print('Start running visualize script')

    # get args
    parser = argparse.ArgumentParser(description='Getting features')
    parser.add_argument('--study', type=str, help='cancer study abbreviation, lowercase')
    parser.add_argument('--project', type=str, help='name of project (spatial_GBM_pred, TCGA-GBM or PESO)')
    parser.add_argument('--gene_names', type=str, help='name of genes to visualize, separated by commas. if you want all the predicted genes, pass "all" ')
    parser.add_argument('--wsi_file_name', type=str, help='wsi filename')
    parser.add_argument('--save_folder', type=str, help='destination folder')
    parser.add_argument('--num_folds', type=int, help='num folds to aggregate over', default=5)
    args = parser.parse_args()

    # general 
    study = args.study

    checkpoint = './vit_exp_log2FPKM-UQ_cor_stop/TCGA_pretrain_no_breast/'+ study + '/'
    obj = read_pickle(checkpoint + 'test_results.pkl')[0]
    gene_ids = obj['genes']

    stride = 1 
    wsi_file_name = args.wsi_file_name
    project = args.project 
    save_path = './visualizations/' + project +'/' + args.save_folder + '/' + args.wsi_file_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.gene_names != 'all':
        # get gene names 
        if '.npy' in args.gene_names:
            gene_names = np.load(args.gene_names,allow_pickle=True)
        else:
            gene_names = args.gene_names.split(",")
    else:
        gene_names = gene_ids
        
    # prepare and load WSI
    if 'TCGA' in wsi_file_name:
        slide_path = './data/Roche-TCGA/'+project+'/'
        mask_path = './data/Roche-TCGA/'+project+'_Masks/'
        mask = np.load(mask_path+wsi_file_name.replace('.svs', '')+'/'+'mask.npy')
        manual_resize = None # nr of um/px can be read from slide properties
    elif project == 'spatial_GBM_pred':
        slide_path = './data/Spatial_GBM/pyramid/'
        mask_path = './data/Spatial_GBM/masks/'
        mask = np.load(mask_path+wsi_file_name.replace('.tif', '.npy'))
        px_df = pd.read_csv('./data/Spatial_Heiland/data/classify/spot_diameter.csv')
        diam = px_df[px_df['slide_id']==wsi_file_name.split('_')[1]+'_T']['pixel_diameter'].values[0]
        um_px = 55/diam # um/px for the WSI
        manual_resize = 0.5/um_px
    else:
        print('please provide correct file name format (containing "TCGA" or "pds") or correct project id ("IvyGap" or "spatial_GBM_pred")')
        exit()

    # slide, patch size
    slide = openslide.OpenSlide(slide_path+wsi_file_name)
    patch_size = 256 # at 20x (0.5um pp)

    # calculate patch size in original image (coordinates are at level 0 for openslide) and in mask
    downsample_factor = int(slide.dimensions[0]/mask.shape[0]) # mask downsample factor

    if manual_resize == None:
        resize_factor = float(slide.properties.get('aperio.AppMag',20)) / 20.0
    else:
        resize_factor = manual_resize

    patch_size_resized = int(resize_factor*patch_size)  
    patch_size_in_mask = int(patch_size_resized/downsample_factor)

    # get valid coordinates (that have tissue)
    valid_idx = []
    mask = (np.transpose(mask, axes=[1,0]))*1
    for col in range(0, slide.dimensions[0]-patch_size_resized, patch_size_resized): #slide.dimensions is (width, height)
        for row in range(0, slide.dimensions[1]-patch_size_resized, patch_size_resized):

            row_downs = int(row/downsample_factor)
            col_downs = int(col/downsample_factor)

            patch_in_mask = mask[row_downs:row_downs+patch_size_in_mask,col_downs:col_downs+patch_size_in_mask]
            patch_in_mask = binary_dilation(patch_in_mask, iterations=3)

            if patch_in_mask.sum() >= (BACKGROUND_THRESHOLD * patch_in_mask.size): 
                # keep patch
                valid_idx.append((col, row))

    # dataframe which contains coordinates of valid patches
    df = pd.DataFrame(valid_idx, columns=['xcoord', 'ycoord'])
    # rescale coordinates to (0,0) and step size 1
    df['xcoord_tf'] = ((df['xcoord']-min(df['xcoord']))/patch_size_resized).astype(int)
    df['ycoord_tf'] = ((df['ycoord']-min(df['ycoord']))/patch_size_resized).astype(int)

    print('Got dataframe of valid tiles')

    # transforms for resnet
    transforms_ = transforms.Compose([
        transforms.Resize((256,265)),
        transforms.CenterCrop((256,265)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # init resnet feature extractor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet50 = resnet50(pretrained=True).to(device)
    resnet50.eval()

    res_df = df.copy(deep=True)

    for fold in range(args.num_folds):

        fold_ckpt = checkpoint + 'model_best_' + str(fold) + '.pt'
        if (fold == 0):
            fold_ckpt = fold_ckpt.replace('_0','')

        model = ViT(num_outputs=len(gene_ids), dim=2048, depth=6, heads=16, mlp_dim=2048, dim_head = 64)
        model.load_state_dict(torch.load(fold_ckpt, map_location=torch.device(device)))
        model = model.to(device)
        model.eval()

        # get indices of requested genes
        inds_gene_of_interest = []
        for gene_name in gene_names:
            try:
                inds_gene_of_interest.append(gene_ids.index(gene_name))
            except:
                print('gene not in predicted values '+gene_name)
        
        # get visualization
        preds = sliding_window_method(df=df, patch_size_resized=patch_size_resized, 
                                        resnet50=resnet50, model=model, 
                                        inds_gene_of_interest=inds_gene_of_interest, stride=stride,
                                        device=device)

        for ind_gene in inds_gene_of_interest:
            res_df[gene_ids[ind_gene] + '_' + str(fold)] = res_df.index.map(preds[ind_gene])

    for ind_gene in inds_gene_of_interest:
        res_df[gene_ids[ind_gene]] = res_df[[gene_ids[ind_gene] + '_' + str(i) for i in range(args.num_folds)]].mean(axis=1)

    save_name = save_path + 'stride-' + str(stride) + '.csv'
    res_df.to_csv(save_name)

    print('Done')

    

    
