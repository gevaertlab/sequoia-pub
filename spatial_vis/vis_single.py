import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import numpy as np
from scipy.stats import percentileofscore
import os
from tqdm import tqdm
import math
import seaborn as sns

def score2percentile(score, ref):
    if np.isnan(score):
        return score # deal with nans in visualization (set to black)
    percentile = percentileofscore(ref, score)
    return percentile

if __name__=='__main__':
    src_path = '/oak/stanford/groups/ogevaert/data/Gen-Pred/visualizations/TCGA-PRAD/'
    folder = 'endo_fibro_lum_rand/'
    draw_heatmaps = False
    all_genes = np.load('/oak/stanford/groups/ogevaert/data/Gen-Pred/gene_ids/prad_experiments/endo_fibro_lum_rand.npy', allow_pickle=True)

    slide_names = os.listdir(src_path + folder)
    slide_names = [i for i in slide_names if i not in ['corr_maps', 'spatial_maps']]
    all_corr_dfs = []

    dests = [src_path + folder + '/corr_maps/', src_path + folder + '/spatial_maps/']
    for dest in dests:
        if not os.path.exists(dest):
            os.makedirs(dest)

    endo = np.load('/oak/stanford/groups/ogevaert/data/Gen-Pred/gene_ids/prad_experiments/endo.npy',allow_pickle=True)
    lum = np.load('/oak/stanford/groups/ogevaert/data/Gen-Pred/gene_ids/prad_experiments/luminal.npy',allow_pickle=True)
    fib = np.load('/oak/stanford/groups/ogevaert/data/Gen-Pred/gene_ids/prad_experiments/fibro.npy',allow_pickle=True)
    rand = [i for i in all_genes if i not in list(endo)+list(fib)+list(lum)]
    mapper = {}
    mapper.update(dict.fromkeys(endo, 'red'))
    mapper.update(dict.fromkeys(rand, 'gray'))
    mapper.update(dict.fromkeys(lum, 'blue'))
    mapper.update(dict.fromkeys(fib, 'green'))

    for slide_name in tqdm(slide_names):

        source_path = src_path + folder + '/' + slide_name
        path = source_path + '/stride-1.csv'
        df = pd.read_csv(path)
        all_genes = list(set(all_genes)&set(df.columns))
        df = df.dropna(axis=0, how='any')
        df = df[['xcoord_tf','ycoord_tf']+all_genes]

        # plt.figure(figsize=(10,10))
        # sns.heatmap(df[all_genes].corr(), annot=False, cmap="coolwarm", vmin=-1, vmax=1)
        # plt.savefig(src_path + folder + '/corr_maps/' + slide_name + '.png', bbox_inches='tight')
        
        corrdf = df[all_genes].corr()
        kind = corrdf.columns.map(mapper)
        all_corr_dfs.append(corrdf)

        plt.close()
        plt.figure()
        pl = sns.clustermap(corrdf, row_colors=kind, yticklabels=True, xticklabels=True, figsize=(50,50))
        pl.ax_row_dendrogram.set_visible(False)
        pl.ax_col_dendrogram.set_visible(False)
        plt.savefig(src_path + folder + '/corr_maps/' + slide_name + '_clustered.png', bbox_inches='tight', dpi=300)
        
        if draw_heatmaps:
            max_lim = max(max(df.xcoord_tf), max(df.ycoord_tf))
            print(f'{slide_name} max lim {max_lim}')

            num_rows = 8
            num_cols = 4
            scaling_factor = max(1,math.ceil(max_lim/28))
            fig, axs = plt.subplots(num_rows,num_cols, figsize=(num_cols*scaling_factor,num_rows*scaling_factor),
                                        sharex=True, sharey=True, subplot_kw=dict(adjustable='box'))

            for i in tqdm(range(num_rows)):
                    for j in tqdm(range(num_cols)):

                        try:
                            gene = all_genes[i*num_cols+j]
                            ref = df[gene].values
                            df[gene + '_perc'] = df.apply(lambda row: score2percentile(row[gene], ref), axis=1)
                            
                            # center the image in the middle of the square
                            if i+j == 0:
                                x_padding = int((max_lim-max(df.xcoord_tf))/2)
                                y_padding = int((max_lim-max(df.ycoord_tf))/2)
                                df['xcoord_tf'] += x_padding
                                df['ycoord_tf'] += y_padding

                            axs[i][j].scatter(df['xcoord_tf']*scaling_factor,
                                                df['ycoord_tf']*scaling_factor, 
                                                s=1, 
                                                c=df[gene+'_perc'], vmin=0, vmax=100, cmap='coolwarm')

                            axs[i][j].set_xlim([0,max_lim*scaling_factor])
                            axs[i][j].set_ylim([0,max_lim*scaling_factor])
                            axs[i][j].set_facecolor("#F1EFF0")
                            for p in ['top', 'right', 'bottom', 'left']:
                                axs[i][j].spines[p].set_color('gray') #.set_visible(False)
                                axs[i][j].spines[p].set_linewidth(0.5)
                            axs[i][j].invert_yaxis()
                            axs[i][j].set_aspect('equal')
                            axs[i][j].tick_params(axis='both', which='both', length=0, labelsize=0)
                            axs[i][j].set_ylabel(gene, fontsize=5*scaling_factor)

                        except:
                            fig.delaxes(axs[i][j])
                            continue

            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.savefig(src_path + folder + '/spatial_maps/' + slide_name + '.png', bbox_inches='tight')
    
    sum_df = all_corr_dfs[0]
    for i in range(1,len(all_corr_dfs)): 
        sum_df += all_corr_dfs[i]
    sum_df = sum_df / len(all_corr_dfs)
    
    # plt.figure(figsize=(10,10))
    # sns.heatmap(sum_df, annot=False, cmap="coolwarm", vmin=-1, vmax=1)
    # plt.savefig(src_path + folder + '/corr_maps/total.png',bbox_inches='tight')

    plt.close()
    plt.figure()
    kind = sum_df.columns.map(mapper)
    pl = sns.clustermap(sum_df, row_colors=kind, col_colors=kind, yticklabels=True, xticklabels=True, figsize=(50,50))
    pl.ax_row_dendrogram.set_visible(False)
    pl.ax_col_dendrogram.set_visible(False)
    plt.savefig(src_path + folder + '/corr_maps/total_clustered.png', bbox_inches='tight', dpi=300)
    
    import pdb; pdb.set_trace()


