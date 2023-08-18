import os
import pandas as pd
import numpy as np
import random
import scanpy as sc
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
import openslide
import cv2
import scipy.ndimage as nd
from PIL import Image

from tqdm import tqdm 

def score2percentile(score, ref):
    if np.isnan(score):
        return score # deal with nans in visualization (set to black)
    percentile = percentileofscore(ref, score)
    return percentile

def get_pred_gt_df(csv_path, AnnData_dir, slide_nr, gene, num_tiles=4):
    """ (1) get ground truth dataframe for slide_nr and gene and 
        (2) add this ground truth to prediction df (which is resampled to match grid of prediction)"""

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

    df2 = pd.read_csv(csv_path) # predictions df
    df2 = df2.dropna(axis=0, how='any')
    df2['ground_truth'] = df2.apply(lambda row: get_average2(row['xcoord'], row['ycoord'], df, num_tiles=num_tiles), axis=1)
    df2 = df2.dropna(axis=0, how='any')

    return df2

def get_average2(xcoord, ycoord, df, num_tiles=4): 
    """ function to match grid of ground truth to grid of prediction """

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

def get_corr_emd_df(path, slide_nrs, suff, nr_gt_vals_min=10, metric='corr'):
    """ get dataframe containing correlations/emd between ground truth and prediction across all genes """
    
    all_dfs = []
    genes = None

    for i, slide_nr in enumerate(slide_nrs):
        slide_name = 'HRI_'+str(slide_nr)+'_T.tif'
        csv_path = path + slide_name + '/metrics.csv'

        df = pd.read_csv(csv_path)
        df = df.sort_values(by='gene')
        if i == 0:
            genes = df['gene'].values

        ind = df.loc[df['nr_gt_vals'+suff]<nr_gt_vals_min].index
        df.loc[ind,'emd'+suff] = np.nan
        df.loc[ind,'corr'+suff] = -np.nan
        df.loc[ind,'sens'+suff] = -np.nan

        all_dfs.append(df.rename(columns={'corr'+suff:'corr_'+slide_nr,
                                           'sens'+suff:'sens_'+slide_nr,
                                           'emd'+suff:'emd_'+slide_nr,})[['corr_'+slide_nr,'sens_'+slide_nr,'emd_'+slide_nr]])
    
    tot_df = pd.concat(all_dfs,axis=1).reset_index(drop=True)
    tot_df['gene'] = genes

    total = tot_df.T
    fillval = -1 if metric == 'corr' else 1e10
    total = total.fillna(fillval)
    total.columns = genes
    total_part = total.loc[[col for col in total.index if metric in col]]

    return total_part

def get_topK(row, k, largest_or_smallest):
    if largest_or_smallest == 'largest':
        return row.index.values[np.argpartition(row, -k)[-k:]]
    else:
        return row.index.values[np.argpartition(row, k)[:k]]

# identifying 'blurriness' of tile
# source https://github.com/gerstung-lab/PC-CHiP/blob/master/inception/preprocess/imgconvert.py
def getGradientMagnitude(im):
    "Get magnitude of gradient for given image"
    im=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ddepth = cv2.CV_32F
    dx = cv2.Sobel(im, ddepth, 1, 0)
    dy = cv2.Sobel(im, ddepth, 0, 1)
    dxabs = cv2.convertScaleAbs(dx)
    dyabs = cv2.convertScaleAbs(dy)
    mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
    return mag

def identify_blurred(tile_arr, amt=15, perc=0.5, perc_white=0.4, size_px=256): #1: tile blurred
    grad = getGradientMagnitude(tile_arr)
    unique, counts = np.unique(grad, return_counts=True)
    if (counts[np.argwhere(unique<=amt)].sum() < size_px*size_px*perc) == False:
        return 1
    
    sought = [150,150,150]
    amt_white  = np.count_nonzero(np.all(tile_arr>sought,axis=2))
    if amt_white/tile_arr.shape[0]**2 > perc_white:
        return 1
    
    return 0

# identify percentage of black in tile
def blackish_percent(img, threshold=(100, 100, 100)):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    t = threshold
    mask = (r < t[0]) & (g < t[1]) & (b < t[2])
    mask = nd.gaussian_filter(mask, sigma=(1, 1), order=0)
    
    percentage = np.mean(mask)
    return percentage

if __name__=='__main__':
    
    preds_path = f'/oak/stanford/groups/ogevaert/data/Gen-Pred/visualizations/spatial_GBM_pred/pretrain_no_breast/vit_best_500genes_vit/'
    res_path = f'/oak/stanford/groups/ogevaert/data/Gen-Pred/visualizations/comparisons/vit_best_500genes_vit/'
    AnnData_dir = "/oak/stanford/groups/ogevaert/data/Spatial_Heiland/data/AnnDataObject/raw"
    slide_path = '/oak/stanford/groups/ogevaert/data/Spatial_GBM/pyramid/'
    dest_path = '/oak/stanford/groups/ogevaert/data/Gen-Pred/visualizations/comparisons/'
    px_df = pd.read_csv('/oak/stanford/groups/ogevaert/data/Spatial_Heiland/data/classify/spot_diameter.csv')
    slide_nrs = ["242", "243", "248", "251", "255",
                 "259", "260", "262", "265", "266",
                 "268", "269", "275", "296", 
                 "313", "334", "270", "304"] 
    
    # set variables
    filtering = True
    across_slides = True # if set to true, plot best genes per slide, otherwise plot genes that work well across slides
    manual_slides = True # whether we use a predefined list of slides and genes
    ext = 'png'
    metric = 'corr' # emd or corr
    norm_by = 'num_tiles'
    metric_asc = False
    largest_or_smallest = 'largest' if metric == 'corr' else 'smallest'
    suff = '_filt' if filtering else ''
    nr_gt_vals_min = 50

    # get dataframe with metrics
    corr_df = get_corr_emd_df(path=res_path, slide_nrs=slide_nrs, metric=metric, suff=suff, nr_gt_vals_min=nr_gt_vals_min)

    if (metric == 'emd') and (norm_by != None):
        # for normalizing emd 
        area_info = pd.read_csv('/oak/stanford/groups/ogevaert/data/Gen-Pred/visualizations/spatial_GBM_pred/slide_info.txt', sep='\t')
        area_info = area_info.drop_duplicates()
        area_info['id'] = 'emd_' + area_info['slide'].str.split('_').str[1]
        norm_by = 'num_tiles' # or area or None

        for id_ in corr_df.index.values:
            corr_df.loc[id_] = corr_df.loc[id_]/np.sqrt(area_info[area_info['id']==id_][norm_by].values[0])

    corr_df_subset = (corr_df>0.15) if metric == 'corr' else (corr_df<0.15)

    if across_slides:
        genes = corr_df_subset.sum(axis=0).sort_values(ascending=metric_asc).head(10).index # genes that have good visual predictions across many slides 
        slides = corr_df_subset.index[(corr_df_subset[genes].sum(axis=1)>=5)] # slides where at least 5 of those genes have a good visual
    
        if manual_slides:
            genes = ['SARAF', 'COL6A1']
            slides = ['emd_262', 'emd_266', 'emd_269', 'emd_275', 'emd_304']
    else:
        corr_df = corr_df.iloc[:9]
        genes = corr_df.apply(lambda row: get_topK(row, 5, largest_or_smallest), axis=1).values
        slides = corr_df.index

        if manual_slides:
            slides = [['266', '269', '275', '270', '262'], ['266', '269', '275', '270', '262'], ['304', '243', '251', '255', '260'], ['304', '243', '251', '255', '260']]
            genes = [['CKAP4', 'TYMS', 'AQP4', 'CYB5B', 'SARAF'], ['COL9A3', 'SMC4', 'RAMP1', 'BEX3', 'COL6A1'], ['MDH2', 'COL1A1', 'FHL1', 'CKLF', 'CA2'], ['CD9', 'ABI1', 'ARHGEF2', 'STK24', 'RDH11']]

    print(slides)
    print(genes)
    
    if not across_slides and manual_slides:
        slide_nrs = np.unique(slides)
    else:
        slide_nrs = [s.split('_')[1] for s in slides]
        slides_opens = [openslide.OpenSlide(slide_path+f'HRI_{s}_T.tif') for s in slide_nrs]
    
    # get max and min coords to coordinate ax lims
    max_x = 0
    max_y = 0
    for slide_nr in slide_nrs:
        df = pd.read_csv(preds_path + f'HRI_{slide_nr}_T.tif' + '/stride-1.csv')
        if max(df.xcoord_tf) > max_x:
            max_x = max(df.xcoord_tf)
        if max(df.ycoord_tf) > max_y:
            max_y = max(df.ycoord_tf)
    max_lim = max(max_x, max_y)

    if not across_slides:
        if manual_slides:

            fig_height = len(slides)*10
            fig_width = (len(genes[0])*2)*10
            fig, axs = plt.subplots(len(slides),len(genes[0])*2, figsize=(fig_width,fig_height),
                                    sharex=True, sharey=True, subplot_kw=dict(adjustable='box'))
            scaling_factor = 2
            for i in tqdm(range(len(slides))):
                for j in tqdm(range(len(genes[0]))):

                    slide_nr = slides[i][j]
                    slide_name = f'HRI_{slide_nr}_T.tif'
                    gene = genes[i][j]

                    # get prediction and ground truth
                    df = get_pred_gt_df(preds_path + slide_name + '/stride-1.csv', AnnData_dir, slide_nr, gene)
                    ref = df[gene].values
                    df[gene + '_perc'] = df.apply(lambda row: score2percentile(row[gene], ref), axis=1)

                    ref2 = df['ground_truth'].values
                    df['ground_truth'] = df.apply(lambda row: score2percentile(row['ground_truth'], ref2), axis=1)
                    if filtering:
                        df['ground_truth_filt'] = df.apply(lambda row: median_filter(df, 'ground_truth', row['xcoord_tf'], row['ycoord_tf'], 1), axis=1)
                    
                    # padding to scatterplot to center the image
                    x_padding = int((max_lim-max(df.xcoord_tf))/2)
                    y_padding = int((max_lim-max(df.ycoord_tf))/2)
                    df['xcoord_tf'] += x_padding
                    df['ycoord_tf'] += y_padding

                    axs[i][2*j].scatter(df['xcoord_tf']*scaling_factor,df['ycoord_tf']*scaling_factor, s=110, c=df[gene+'_perc'], vmin=0, vmax=100, cmap='coolwarm')
                    axs[i][2*j+1].scatter(df['xcoord_tf']*scaling_factor, df['ycoord_tf']*scaling_factor, s=110, c=df['ground_truth'+suff], vmin=0, vmax=100, cmap='coolwarm')

                    for ind in [2*j, 2*j+1]:

                        axs[i][ind].set_xlim([0,max_lim*scaling_factor])
                        axs[i][ind].set_ylim([0,max_lim*scaling_factor])
                        axs[i][ind].set_xticks([i for i in range(max_lim*scaling_factor)])
                        axs[i][ind].set_yticks([i for i in range(max_lim*scaling_factor)])

                        axs[i][ind].set_facecolor("#F1EFF0")
                        for p in ['top', 'right', 'bottom', 'left']:
                            axs[i][ind].spines[p].set_color('gray') #.set_visible(False)
                            axs[i][ind].spines[p].set_linewidth(2)

                        axs[i][ind].invert_yaxis()
                        axs[i][ind].set_aspect('equal')

                        axs[i][ind].tick_params(axis='both', which='both', length=0, labelsize=0)

                    if i % 2 == 0:
                        axs[i][2*j].set_title(f'pred slide {slide_nr}', fontsize=50)
                        axs[i][2*j+1].set_title(f'true slide {slide_nr}', fontsize=50)

                    axs[i][2*j].set_ylabel(gene, fontsize=50)

            plt.subplots_adjust(wspace=0.15, hspace=0.15)
            plt.savefig(f'{dest_path}/{metric}_within_slides_manual.png', bbox_inches='tight')

        else:
            fig_height = len(genes[0])*2
            fig_width = (len(slides)*3)*2
            fig, axs = plt.subplots(len(genes[0]),len(slides)*3, figsize=(fig_width,fig_height), 
                                    sharex=True, sharey=True, subplot_kw=dict(adjustable='box'))
            scaling_factor = 7
            blue = [134, 153, 230]
            red = [206, 87, 72]
            top, bottom, left, right = [10]*4
            for i in tqdm(range(len(slides))): 
                for j in tqdm(range(len(genes[i]))):
                    slide_nr = slides[i].split('_')[1]
                    slide_name = f'HRI_{slide_nr}_T.tif'
                    gene = genes[i][j]
                    
                    # get prediction and ground truth
                    df = get_pred_gt_df(preds_path + slide_name + '/stride-1.csv', AnnData_dir, slide_nr, gene)
                    ref = df[gene].values
                    df[gene + '_perc'] = df.apply(lambda row: score2percentile(row[gene], ref), axis=1)

                    ref2 = df['ground_truth'].values
                    df['ground_truth'] = df.apply(lambda row: score2percentile(row['ground_truth'], ref2), axis=1)
                    if filtering:
                        df['ground_truth_filt'] = df.apply(lambda row: median_filter(df, 'ground_truth', row['xcoord_tf'], row['ycoord_tf'], 1), axis=1)
                    
                    # get tiles with highest/lowest pred
                    diam = px_df[px_df['slide_id']==slide_nr+'_T']['pixel_diameter'].values[0]
                    um_px = 55/diam
                    manual_resize = 0.5/um_px
                    patch_size_resized = int(manual_resize*256)
                    max_min_patches = []
                    num_patches_search = 15
                    num_patches = 2
                    for asc in [False, True]:
                        xcoords =  df.sort_values(by=gene, ascending=asc).xcoord.values[:num_patches_search]
                        ycoords =  df.sort_values(by=gene, ascending=asc).ycoord.values[:num_patches_search]

                        # randomly shuffle these patches
                        combined = list(zip(xcoords, ycoords))
                        random.shuffle(combined)
                        xcoords, ycoords = zip(*combined)

                        patches = []
                        for t in range(num_patches_search):
                            if len(patches) == (num_patches*2): # times two because padding is added as well
                                break
                            patch_ = slides_opens[i].read_region((xcoords[t], ycoords[t]), 0, (patch_size_resized, patch_size_resized))
                            patch_ = patch_.resize((256,256),Image.Resampling.LANCZOS)
                            
                            if not identify_blurred(np.asarray(patch_.convert('RGB'))):
                                color = red if asc == False else blue
                                patch = cv2.copyMakeBorder(np.asarray(patch_.convert('RGB')), top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
                                patches.append(patch)
                                patches.append((np.ones((patch.shape[0],3,3))*255).astype(np.uint8)) # white padding
                        if len(patches) < (num_patches*2):
                            for _ in range((num_patches*2 - len(patches))//2):
                                patches.append((np.ones(patch.shape)*255).astype(np.uint8)) # white patch
                                patches.append((np.ones((patch.shape[0],3,3))*255).astype(np.uint8)) # white padding

                        max_min_patches.append(np.concatenate(patches, axis=1))
                        max_min_patches.append((np.ones((3,max_min_patches[0].shape[1],3))*255).astype(np.uint8)) # white padding
                    
                    # rescale patches to fit box width of scatterplots
                    patch_grid = np.concatenate(max_min_patches, axis=0)
                    im = Image.fromarray(patch_grid)
                    im = im.resize((max_lim*scaling_factor,max_lim*scaling_factor),Image.Resampling.LANCZOS)
                    patch_grid = np.asarray(im)

                    for ind in [3*i, 3*i+1]:
                        axs[j][ind].set_xlim([0,max_lim*scaling_factor])
                        axs[j][ind].set_ylim([0,max_lim*scaling_factor])
                        axs[j][ind].set_xticks([i for i in range(max_lim*scaling_factor)])
                        axs[j][ind].set_yticks([i for i in range(max_lim*scaling_factor)])

                        axs[j][ind].invert_yaxis()
                        axs[j][ind].set_aspect('equal')

                        for p in ['top', 'right', 'bottom', 'left']:
                            axs[j][ind].spines[p].set_color('gray') #.set_visible(False)
                            axs[j][ind].spines[p].set_linewidth(0.5)

                        axs[j][ind].set_xticks([])
                        axs[j][ind].set_yticks([])
                        axs[j][ind].set_facecolor("#F1EFF0")
                        if ind == 3*i:
                            axs[j][ind].set_title(f'slide {slide_nr}, {gene}', fontsize=8, loc='left')

                    # padding to scatterplot to center the image
                    x_padding = int((max_lim-max(df.xcoord_tf))/2)
                    y_padding = int((max_lim-max(df.ycoord_tf))/2)
                    df['xcoord_tf'] += x_padding
                    df['ycoord_tf'] += y_padding

                    # plot 
                    size_ = 2
                    axs[j][3*i].scatter(df['xcoord_tf']*scaling_factor, df['ycoord_tf']*scaling_factor, s=size_, c=df[gene+'_perc'], vmin=0, vmax=100, cmap='coolwarm')
                    axs[j][3*i+1].scatter(df['xcoord_tf']*scaling_factor, df['ycoord_tf']*scaling_factor, s=size_, c=df['ground_truth'+suff], vmin=0, vmax=100, cmap='coolwarm')

                    axs[j][3*i+2].imshow(patch_grid)
                    axs[j][3*i+2].axis('off')

            plt.subplots_adjust(wspace=0.1, hspace=0.2)
            plt.savefig(f'{dest_path}/{metric}_within_slides_0_min{nr_gt_vals_min}.{ext}', bbox_inches='tight',dpi=300)
            import pdb; pdb.set_trace()

    else:
        fig_height = len(genes)*10
        fig_width = (len(slides)*2)*10
        fig, axs = plt.subplots(len(genes),len(slides)*2, figsize=(fig_width,fig_height),
                                sharex=True, sharey=True, subplot_kw=dict(adjustable='box'))
        scaling_factor = 2
        for i in tqdm(range(len(genes))):
            for j in tqdm(range(len(slides))):

                slide_nr = slides[j].split('_')[1]
                slide_name = f'HRI_{slide_nr}_T.tif'
                gene = genes[i]

                # get prediction and ground truth
                df = get_pred_gt_df(preds_path + slide_name + '/stride-1.csv', AnnData_dir, slide_nr, gene)
                ref = df[gene].values
                df[gene + '_perc'] = df.apply(lambda row: score2percentile(row[gene], ref), axis=1)

                ref2 = df['ground_truth'].values
                df['ground_truth'] = df.apply(lambda row: score2percentile(row['ground_truth'], ref2), axis=1)
                if filtering:
                    df['ground_truth_filt'] = df.apply(lambda row: median_filter(df, 'ground_truth', row['xcoord_tf'], row['ycoord_tf'], 1), axis=1)
                
                # padding to scatterplot to center the image
                x_padding = int((max_lim-max(df.xcoord_tf))/2)
                y_padding = int((max_lim-max(df.ycoord_tf))/2)
                df['xcoord_tf'] += x_padding
                df['ycoord_tf'] += y_padding

                axs[i][2*j].scatter(df['xcoord_tf']*scaling_factor,df['ycoord_tf']*scaling_factor, s=110, c=df[gene+'_perc'], vmin=0, vmax=100, cmap='coolwarm')
                axs[i][2*j+1].scatter(df['xcoord_tf']*scaling_factor, df['ycoord_tf']*scaling_factor, s=110, c=df['ground_truth'+suff], vmin=0, vmax=100, cmap='coolwarm')

                for ind in [2*j, 2*j+1]:

                    axs[i][ind].set_xlim([0,max_lim*scaling_factor])
                    axs[i][ind].set_ylim([0,max_lim*scaling_factor])
                    axs[i][ind].set_xticks([i for i in range(max_lim*scaling_factor)])
                    axs[i][ind].set_yticks([i for i in range(max_lim*scaling_factor)])

                    axs[i][ind].set_facecolor("#F1EFF0")
                    for p in ['top', 'right', 'bottom', 'left']:
                        axs[i][ind].spines[p].set_color('gray') #.set_visible(False)
                        axs[i][ind].spines[p].set_linewidth(2)

                    axs[i][ind].invert_yaxis()
                    axs[i][ind].set_aspect('equal')

                    axs[i][ind].tick_params(axis='both', which='both', length=0, labelsize=0)

                if i == 0:
                    axs[i][2*j].set_title(f'pred slide {slide_nr}', fontsize=50)
                    axs[i][2*j+1].set_title(f'true slide {slide_nr}', fontsize=50)

                if j % (len(slides)*2) == 0:
                    axs[i][2*j].set_ylabel(genes[i], fontsize=50)

        plt.subplots_adjust(wspace=0.15, hspace=0.15)
        plt.savefig(f'{dest_path}/{metric}_across_slides.png', bbox_inches='tight')