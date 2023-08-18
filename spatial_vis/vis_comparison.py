import os

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from matplotlib import rcParams

import scanpy as sc
import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr 
import cv2
import ot
import gc
from tqdm import tqdm

from scipy.stats import percentileofscore

def score2percentile(score, ref):
    if np.isnan(score):
        return score # deal with nans in visualization (set to black)
    percentile = percentileofscore(ref, score)
    return percentile

def get_average(xcoord_tf, ycoord_tf, df, resize_factor, num_tiles, offset):
    """"
    for xcoord_tf and ycoord_tf in df2, get the corresponding
    average gene expression value from df, where the resize_factor 
    is equal to factor (to match the two resolutions)
    """

    xcoord_tf_matched = xcoord_tf * resize_factor
    ycoord_tf_matched = ycoord_tf * resize_factor
    
    # note: the WSI coordinates are for patch starting at left upper corner
    window = df[ (df['x_tf'] >= xcoord_tf_matched - offset) & 
                 (df['y_tf'] >= ycoord_tf_matched - offset) &
                 (df['x_tf'] < xcoord_tf_matched + num_tiles * resize_factor + offset) & 
                 (df['y_tf'] < ycoord_tf_matched + num_tiles * resize_factor + offset) ]

    if len(window['gene_expr'].values) > 0:
        #print(len(window['gene_expr'].values) )
        return np.mean(window['gene_expr'].values)

    else:
        #import pdb; pdb.set_trace()
        return np.nan

def get_average2(xcoord, ycoord, df, num_tiles): 

    distances_x = np.power(df['x'] - xcoord, 2).values
    distances_y = np.power(df['y'] - ycoord, 2).values
    distances = np.sqrt(distances_x + distances_y)
    closest_samples = sorted(range(len(distances)), key = lambda sub: distances[sub])[:num_tiles]

    gene_expr_vals = []
    for i in closest_samples:
        gene_expr_vals.append(df.iloc[i]['gene_expr'])

    return np.mean(gene_expr_vals)


def median_filter(df, col, xcoord, ycoord, num_neighbors):
    window = df[ (df['xcoord_tf'] >= xcoord - num_neighbors) & 
                 (df['ycoord_tf'] >= ycoord - num_neighbors) &
                 (df['xcoord_tf'] <= xcoord + num_neighbors) & 
                 (df['ycoord_tf'] <= ycoord + num_neighbors) ]

    full_window_size = (num_neighbors*2+1)**2
    if window.shape[0] > full_window_size/2:
        return np.median(window[col].values)
    
    return df[(df['xcoord_tf'] == xcoord) & (df['ycoord_tf'] == ycoord)][col].values[0]


def img_to_sig(arr):
    """Convert a 2D array to a signature for cv2.EMD"""
    
    # cv2.EMD requires single-precision, floating-point input
    sig = np.empty((arr.size, 3), dtype=np.float32)
    count = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            sig[count] = np.array([arr[i,j], i, j])
            count += 1
    return sig

    
def calculate_emd(arr1, arr2, norm=False):
    assert arr1.shape == arr2.shape, "please provide consistent shapes"
    assert len(arr1.shape) == 2, "please give nxm matrix format"

    if (not np.any(arr1)) and (not np.any(arr2)): # if both are totally 0 then the EMD is 0
        return 0

    # if one of the two maps is totally zero and the other is not, the EMD is not defined
    # in that case we return NaN
    if not np.any(arr1):
        return np.nan
    if not np.any(arr2):
        return np.nan
        
    arr1 = arr1 / np.sum(arr1)
    arr2 = arr2 / np.sum(arr2)

    sig1 = img_to_sig(arr1)
    sig2 = img_to_sig(arr2)
    dist, _, _ = cv2.EMD(sig1, sig2, cv2.DIST_L2)

    if norm: 
        dist = dist / np.sqrt(arr1.shape[0]*arr2.shape[0])
    return dist


def fill_arr(arr, x, y, val):
    arr[x,y] = val

if __name__=='__main__':

    # get args
    parser = argparse.ArgumentParser(description='Getting features')
    parser.add_argument('--slide_nr', type=str, help='slide nr for which to run script')
    parser.add_argument('--pred_folder', type=str, help='folder with predictions to visualize')
    parser.add_argument('--save_folder', type=str, help='where to save results')
    parser.add_argument('--gene_names', type=str, help='name of genes to visualize (separated by comma) or path to npy array containing gene names')
    args = parser.parse_args()

    slide_nr = args.slide_nr
    preds_path = f'/oak/stanford/groups/ogevaert/data/Gen-Pred/visualizations/spatial_GBM_pred/{args.pred_folder}/'
    dest_path = f'/oak/stanford/groups/ogevaert/data/Gen-Pred/visualizations/comparisons/{args.save_folder}/'

    slide_name = 'HRI_'+str(slide_nr)+'_T.tif'
    print(slide_name)
    csv_path = preds_path + slide_name + '/stride-1.csv'
    
    gene_names = args.gene_names
    if '.npy' in gene_names:
        genes = np.load(gene_names, allow_pickle=True)
    else:
        genes = gene_names.split(",")

    dest_path += slide_name + '/'
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    num_tiles = 4 # how many tiles in ground truth are equal to one tile in prediction (same for all of spatial gbm because prediction resolution is patch of 256x256 at 0.5um pp)

    correlations = {}
    pvals = {}
    sens_vals = {}
    emds = {}
    nr_gt_vals = {}

    # after converting the ground truth and prediction to 0-100 and applying median filtering to ground truth
    correlations_filt = {}
    pvals_filt = {}
    sens_vals_filt = {}
    emds_filt = {}
    nr_gt_vals_filt = {}

    rcParams['font.family'] = 'sans-serif'
    fig_resize = 1

    for i_, gene in tqdm(enumerate(genes)):

        try:

            # get ground truth data
            AnnData_dir = "/oak/stanford/groups/ogevaert/data/Spatial_Heiland/data/AnnDataObject/raw"
            adata = sc.read_h5ad(os.path.join(AnnData_dir, f'{slide_nr}_T.h5ad'))
            sc.pp.normalize_total(adata, inplace=True)
            sc.pp.log1p(adata)
            sc.pp.scale(adata)

            adata_subset = adata[:,gene]
            coords = adata_subset.obs[['x', 'y']].values
            gene_expr = np.asarray(adata_subset.X).flatten()

            df = pd.DataFrame(coords, columns=['x', 'y'])
            df['gene_expr'] = gene_expr
            df['x_tf'] = (df['x']-min(df['x'])).astype(int) # transform coordinates to regular grid
            df['y_tf'] = (df['y']-min(df['y'])).astype(int)

            # transform ground truth to same resolution
            df2 = pd.read_csv(csv_path)
            df2 = df2.dropna(axis=0, how='any')
            df2['ground_truth'] = df2.apply(lambda row: get_average2(row['xcoord'], row['ycoord'], df, num_tiles=num_tiles), axis=1)
            df2 = df2.dropna(axis=0, how='any')

            # transform to percentile and perform median filtering 
            # (med filtering only for ground truth, gene prediction is already smooth because of sliding window method)
            ref = df2['ground_truth'].values
            df2['ground_truth_perc'] = df2.apply(lambda row: score2percentile(row['ground_truth'], ref), axis=1)
            df2['ground_truth_filt'] = df2.apply(lambda row: median_filter(df2, 'ground_truth_perc', row['xcoord_tf'], row['ycoord_tf'], 1), axis=1)
            
            #df2['ground_truth_filt'] = df2.apply(lambda row: median_filter(df2, 'ground_truth', row['xcoord_tf'], row['ycoord_tf'], 1), axis=1)
            #ref = df2['ground_truth_filt'].values
            #df2['ground_truth_filt'] = df2.apply(lambda row: score2percentile(row['ground_truth_filt'], ref), axis=1)

            ref2 = df2[gene].values
            df2[gene + '_filt'] = df2.apply(lambda row: score2percentile(row[gene], ref2), axis=1) 
            #df2[gene + '_filt'] = df2[gene]
            
            for i, gt_col, gene_col in zip(range(2), ['ground_truth', 'ground_truth_filt'], [gene, gene + '_filt']):

                # binarize and calculate sensitivity
                quant = df2[gt_col].quantile(0.75)
                quant2 = df2[gene_col].quantile(0.75)
                df2['gt_binary'] = df2[gt_col].apply(lambda x: 1 if x >= quant else 0)
                df2[gene_col+'_binary'] = df2[gene_col].apply(lambda x: 1 if x >= quant2 else 0)
                tp = df2[(df2['gt_binary']==1)&(df2[gene_col+'_binary']==1)]
                fn = df2[(df2['gt_binary']==1)&(df2[gene_col+'_binary']==0)]
                sens = tp.shape[0]/(tp.shape[0]+fn.shape[0]+1e-10)

                # calculate correlation
                coef, p = pearsonr(df2[gt_col], df2[gene_col])

                # get EMD
                max_x = max(df2.xcoord_tf)
                max_y = max(df2.ycoord_tf)
                arr0 = np.zeros((max_x+1, max_y+1))
                df2.apply(lambda row: fill_arr(arr0, row['xcoord_tf'].astype(int),row['ycoord_tf'].astype(int), row[gene_col]),axis=1)
                arr1 = np.zeros((max_x+1, max_y+1))
                df2.apply(lambda row: fill_arr(arr1, row['xcoord_tf'].astype(int),row['ycoord_tf'].astype(int), row[gt_col]),axis=1)
                
                arr0 = arr0 + np.abs(np.min(arr0))
                arr1 = arr1 + np.abs(np.min(arr1))
                emd = calculate_emd(arr0, arr1, norm=False)

                if i == 0:
                    sens_vals[gene] = sens
                    correlations[gene] = coef
                    pvals[gene] = p
                    emds[gene] = emd
                else:
                    sens_vals_filt[gene] = sens
                    correlations_filt[gene] = coef
                    pvals_filt[gene] = p
                    emds_filt[gene] = emd

            ##### enough to only run this once
            # # write the area, nr of tiles per slide to a file so normalization can be done afterwards if desired
            # if i_ == 0:
            #     filename = "/oak/stanford/groups/ogevaert/data/Gen-Pred/visualizations/spatial_GBM_pred/slide_info.txt"
            #     with open(filename, 'a') as file:
            #         file.write(f"{slide_name} \t {arr0.shape[0]*arr1.shape[0]} \t {df2.shape[0]} \n")

            # also write per slide, per gene, the number of unique values in the ground truth to file to detect any artefacts if needed
            nr_gt_vals[gene] = len(np.unique(df2['ground_truth'].values))
            nr_gt_vals_filt[gene] = len(np.unique(df2['ground_truth_filt'].values))

        except Exception as e:
            print(e)
            print(gene)

    gc.collect()


    corr_df = pd.DataFrame(correlations.items(), columns=['gene', 'corr'])
    sens_df = pd.DataFrame(sens_vals.items(), columns=['gene', 'sens'])
    pval_df = pd.DataFrame(pvals.items(), columns=['gene', 'pval'])
    emd_df = pd.DataFrame(emds.items(), columns=['gene', 'emd'])
    nr_gt_df = pd.DataFrame(nr_gt_vals.items(), columns=['gene', 'nr_gt_vals'])

    corr_df_filt = pd.DataFrame(correlations_filt.items(), columns=['gene', 'corr_filt'])
    sens_df_filt = pd.DataFrame(sens_vals_filt.items(), columns=['gene', 'sens_filt'])
    pval_df_filt = pd.DataFrame(pvals_filt.items(), columns=['gene', 'pval_filt'])
    emd_df_filt = pd.DataFrame(emds_filt.items(), columns=['gene', 'emd_filt'])
    nr_gt_df_filt = pd.DataFrame(nr_gt_vals_filt.items(), columns=['gene', 'nr_gt_vals_filt'])

    total_df = pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(
                                                    corr_df, pval_df, on='gene'), 
                                                    sens_df, on='gene'), 
                                                    emd_df, on='gene'), 
                                                    nr_gt_df, on='gene'),
                                                    corr_df_filt, on='gene'),
                                                    sens_df_filt, on='gene'),
                                                    pval_df_filt, on='gene'),
                                                    emd_df_filt, on='gene'),
                                                    nr_gt_df_filt, on='gene')

    total_df.to_csv(dest_path + '/' + 'metrics.csv')
    print('Done')







#     condition = coef > 0.1 #True
        #     if condition:
        #         # visualize
        #         if use_percentile:

        #             ref2 = df2[gene].values
        #             df2[gene+'_perc'] = df2.apply(lambda row: score2percentile(row[gene], ref2), axis=1)

        #             ref3 = df2['ground_truth'].values
        #             df2['ground_truth_perc'] = df2.apply(lambda row: score2percentile(row['ground_truth'], ref3), axis=1)

        #             if median_filtering:
        #                 ref4 = df2['ground_truth_filt'].values
        #                 df2['ground_truth_filt_perc'] = df2.apply(lambda row: score2percentile(row['ground_truth_filt'], ref4), axis=1)
        #                 col = 'ground_truth_filt_perc'
        #             else:
        #                 col = 'ground_truth_perc'

        #             col_gene = gene+'_perc'
        #             vmin = 0
        #             vmax = 100

        #         else:
        #             vmin = min(min(df2[gene].values), min(df['gene_expr'].values))
        #             vmax = max(max(df2[gene].values), max(df['gene_expr'].values))
        #             col = 'ground_truth'
        #             col_gene = gene

        #         f, axarr = plt.subplots(1,2, figsize=((15/fig_resize,5)))

        #         axarr[1].scatter(df2['xcoord_tf']*2, df2['ycoord_tf']*2, c=df2[col], s = 24*fig_resize, vmin=vmin, vmax=vmax)
        #         axarr[0].scatter(df2['xcoord_tf']*2, df2['ycoord_tf']*2, c=df2[col_gene], s = 24*fig_resize, vmin=vmin, vmax=vmax)
                
        #         axarr[0].axis('off')
        #         axarr[1].axis('off')
        #         axarr[0].invert_yaxis()
        #         axarr[1].invert_yaxis()
        #         axarr[0].axis('equal')
        #         axarr[1].axis('equal')

        #         norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        #         mapper = cm.ScalarMappable(norm=norm) 

        #         plt.colorbar(mapper, ax=axarr.ravel().tolist(), shrink=0.95)
        #         plt.suptitle(gene +', corr %.3f, p = %.3g' %(coef, p), y=0.95)
        #         plt.savefig(dest_path + '/' + gene+'.png', bbox_inches='tight', dpi=300)
        #         plt.close('all')
        #         gc.collect() # clear memory 
