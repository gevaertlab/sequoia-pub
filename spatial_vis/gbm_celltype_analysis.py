import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import numpy as np
from scipy.stats import percentileofscore
import os
from tqdm import tqdm
import math
import seaborn as sns
import matplotlib.colors

def score2percentile(score, ref):
    if np.isnan(score):
        return score # deal with nans in visualization (set to black)
    percentile = percentileofscore(ref, score)
    return percentile

if __name__=='__main__':
    root = '.'
    src_path = root + 'visualizations/spatial_GBM_pred/'
    folder = 'gbm_celltypes/'
    draw_heatmaps = True
    all_genes = np.load(root + 'gene_ids/gbm_experiments/all.npy', allow_pickle=True)

    slide_names = os.listdir(src_path + folder)
    slide_names = [i for i in slide_names if i not in ['corr_maps', 'spatial_maps']]
    all_corr_dfs = []

    dests = [src_path + folder + '/corr_maps/', src_path + folder + '/spatial_maps/']
    for dest in dests:
        if not os.path.exists(dest):
            os.makedirs(dest)

    ac = np.load(root + 'gene_ids/celltypes/AC.npy',allow_pickle=True)
    g1s = np.load(root + 'gene_ids/celltypes/G1S.npy',allow_pickle=True)
    g2m = np.load(root + 'gene_ids/celltypes/G2M.npy',allow_pickle=True)
    mes1 = np.load(root + 'gene_ids/celltypes/MES1.npy',allow_pickle=True)
    mes2 = np.load(root + 'gene_ids/celltypes/MES2.npy',allow_pickle=True)
    npc1 = np.load(root + 'gene_ids/celltypes/NPC1.npy',allow_pickle=True)
    npc2 = np.load(root + 'gene_ids/celltypes/NPC2.npy',allow_pickle=True)
    opc = np.load(root + 'gene_ids/celltypes/OPC.npy',allow_pickle=True)
    mapper = {}

    green = '#CEBC36'
    red = '#CE3649'
    blue = '#3648CE'
    purple = '#36CEBC'

    mapper.update(dict.fromkeys(ac, matplotlib.colors.to_rgb(purple))) # purple
    mapper.update(dict.fromkeys(g1s, matplotlib.colors.to_rgb(red))) # red
    mapper.update(dict.fromkeys(g2m, matplotlib.colors.to_rgb(red)))
    mapper.update(dict.fromkeys(mes1, matplotlib.colors.to_rgb(blue))) #blue
    mapper.update(dict.fromkeys(mes2, matplotlib.colors.to_rgb(blue)))
    mapper.update(dict.fromkeys(npc1, matplotlib.colors.to_rgb(green))) #green
    mapper.update(dict.fromkeys(npc2, matplotlib.colors.to_rgb(green)))
    mapper.update(dict.fromkeys(opc, matplotlib.colors.to_rgb(green)))

    max_lim = 0

    for slide_name in tqdm(slide_names):

        source_path = src_path + folder + '/' + slide_name
        path = source_path + '/stride-1.csv'
        df = pd.read_csv(path)

        df_max = max_lim = max(max(df.xcoord_tf), max(df.ycoord_tf))
        if df_max > max_lim:
            max_lim = df_max

        all_genes = list(set(all_genes)&set(df.columns))
        
        df = df.dropna(axis=0, how='any')
        df = df[['xcoord_tf','ycoord_tf']+all_genes]

        corrdf = df[all_genes].corr()
        kind = corrdf.columns.map(mapper)
        all_corr_dfs.append(corrdf)

        plt.close()
        plt.figure()
        pl = sns.clustermap(corrdf, row_colors=kind, cmap='magma') #, yticklabels=True, xticklabels=True, figsize=(50,50)
        pl.ax_row_dendrogram.set_visible(False)
        pl.ax_col_dendrogram.set_visible(False)
        plt.savefig(src_path + folder + '/corr_maps/' + slide_name + '_clustered.png', bbox_inches='tight', dpi=300)
    
    if draw_heatmaps:

        scaling_factor = 2
        max_lim += scaling_factor*5

        for slide_name in tqdm(slide_names):

            source_path = src_path + folder + '/' + slide_name
            path = source_path + '/stride-1.csv'
            df = pd.read_csv(path)
            all_genes = list(set(all_genes)&set(df.columns))
            df = df.dropna(axis=0, how='any')
            df = df[['xcoord_tf','ycoord_tf']+all_genes]
        
            categories = [ac.tolist(), g1s.tolist()+g2m.tolist(), mes1.tolist()+mes2.tolist(), npc1.tolist()+npc2.tolist()+opc.tolist()]
            labels = ['ac', 'cc', 'mes', 'lin']
            colors = {'ac':purple, 'cc':red, 'mes':blue, 'lin':green}
            
            for j,label in enumerate(labels):
                df[label] = df[[i for i in categories[j] if i in df.columns]].mean(axis=1)
                ref = df[label].values
                df[label + '_perc'] = df.apply(lambda row: score2percentile(row[label], ref), axis=1)

            df['color'] = df[[i+'_perc' for i in labels]].idxmax(axis=1)
            df['color'] = df['color'].str.replace('_perc', '')
            df['color'] = df['color'].map(colors)

            plt.close()
            fig, ax = plt.subplots()
            x_padding = int((max_lim-max(df.xcoord_tf))/2)
            y_padding = int((max_lim-max(df.ycoord_tf))/2)
            df['xcoord_tf'] += x_padding
            df['ycoord_tf'] += y_padding
            
            ax.scatter(df['xcoord_tf']*scaling_factor,
                        df['ycoord_tf']*scaling_factor, 
                        s=17, 
                        c=df['color'])

            ax.set_xlim([0,max_lim*scaling_factor])
            ax.set_ylim([0,max_lim*scaling_factor])
            ax.set_facecolor("#F1EFF0")
            for p in ['top', 'right', 'bottom', 'left']:
                ax.spines[p].set_color('gray') #.set_visible(False)
                ax.spines[p].set_linewidth(1)
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.tick_params(axis='both', which='both', length=0, labelsize=0)

            plt.savefig(src_path + folder + '/spatial_maps/' + slide_name + '.png', bbox_inches='tight', dpi=300)
    
    sum_df = all_corr_dfs[0]
    for i in range(1,len(all_corr_dfs)): 
        sum_df += all_corr_dfs[i]
    sum_df = sum_df / len(all_corr_dfs)
    
    plt.close()
    plt.figure()
    kind = sum_df.columns.map(mapper)
    pl = sns.clustermap(sum_df, row_colors=kind, col_colors=kind, cmap='magma')
    pl.ax_row_dendrogram.set_visible(False)
    pl.ax_col_dendrogram.set_visible(False)
    plt.savefig(src_path + folder + '/corr_maps/total_clustered.png', bbox_inches='tight', dpi=300)
    


