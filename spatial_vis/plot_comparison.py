import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats

def p_to_text(p):
    text = 'ns'
    if (p < 0.05):
        text = '*'
    if (p < 1e-2):
        text = '**'
    if (p < 1e-3):
        text = '***'
    if (p < 1e-4):
        text = '****'

    return text

if __name__=='__main__':

    all_studies = ['vit','he2rna'] 
    proteincoding_only = False
    filtering = True
    nr_gt_vals_min = 10
    
    suff_p = '_proteincoding' if proteincoding_only else ''
    path = f'/oak/stanford/groups/ogevaert/data/Gen-Pred/visualizations/comparisons/'
    suff = '_filt' if filtering else ''
    
    # for normalizing emd 
    area_info = pd.read_csv('/oak/stanford/groups/ogevaert/data/Gen-Pred/visualizations/spatial_GBM_pred/slide_info.txt', sep='\t')
    area_info = area_info.drop_duplicates()
    area_info['id'] = 'emd_' + area_info['slide'].str.split('_').str[1]
    norm_by = 'num_tiles' # or area or None

    slide_nrs = ["242", "243", "248", "251", "255",
                "259", "260", "262", "265", "266",
                "268", "269", "275", "296", 
                "313", "334", "270", "304"] 
    remove_slides = [] #"268", "313", "334", "265"

    sns.set()
    sns.set_style("dark")
    colors = ["#2EC4B6", "#FF9F1C"]
    customPalette = sns.set_palette(sns.color_palette(colors))

    # plot comparison between study X and study Y, where for each study we consider best genes
    dest_folder0 = f'{path}/results{suff_p}/vit_best_comparison/'
    if not os.path.exists(dest_folder0):
        os.makedirs(dest_folder0)

    emds_slide = []
    corrs_slide = []
    sens_slide = []
    for stud in all_studies:
        for slide_nr in slide_nrs:
            slide_name = 'HRI_'+str(slide_nr)+'_T.tif'

            suff_temp = '' #'/archive/' 
            csv_path = f'{path}/{suff_temp}/NEW2_vit_best_500genes_{stud}/{slide_name}/metrics.csv'

            df = pd.read_csv(csv_path)
            df = df[df['nr_gt_vals'+suff]>=nr_gt_vals_min]
            
            if proteincoding_only:
                genedf = pd.read_csv('/oak/stanford/groups/ogevaert/data/Gen-Pred/ref_files/ensembl_gene_map.csv')
                okgenes = genedf[genedf['biotype']=='protein_coding']['hgnc'].values
                df = df[df['gene'].isin(okgenes)]

            if norm_by != None:
                df['emd'+suff] = df['emd'+suff]/np.sqrt(area_info[area_info['id']=='emd_'+str(slide_nr)][norm_by].values[0])

            df['study'] = stud
            df['slide_nr'] = slide_nr

            #import pdb; pdb.set_trace()
            emds_slide += df.sort_values(by='emd'+suff, ascending=True)[['study', 'slide_nr', 'emd'+suff,'gene']].values.tolist()
            corrs_slide += df.sort_values(by='corr'+suff, ascending=False)[['study', 'slide_nr', 'corr'+suff,'gene']].values.tolist()
            sens_slide += df.sort_values(by='sens'+suff, ascending=False)[['study', 'slide_nr', 'sens'+suff,'gene']].values.tolist()
            # for emd in df.sort_values(by='emd'+suff, ascending=True)['emd'+suff].values: #.iloc[:100]
            #     emds_slide.append([stud, slide_nr, emd])
            # for corr in df.sort_values(by='corr'+suff, ascending=False)['corr'+suff].values:
            #     corrs_slide.append([stud, slide_nr, corr])
            # for sens in df.sort_values(by='sens'+suff, ascending=False)['sens'+suff].values:
            #     sens_slide.append([stud, slide_nr, sens])

    emd_df = pd.DataFrame(emds_slide, columns=['study','slide','emd','gene'])
    corr_df = pd.DataFrame(corrs_slide, columns=['study','slide','corr','gene'])
    sens_df = pd.DataFrame(sens_slide, columns=['study','slide','sens','gene'])
    dfs_ = [emd_df, corr_df, sens_df]

    for i, metric in enumerate(['emd', 'corr', 'sens']):
        plt.figure(figsize=(20,5))
        fillval = 0 if ((metric == 'corr') or (metric == 'sens')) else dfs_[i][metric].max()
        dfs_[i] = dfs_[i].fillna(fillval)
        ax = sns.violinplot(x=dfs_[i]['slide'].values, y=dfs_[i][metric].values, hue=dfs_[i]['study'].values, 
                        palette=customPalette, hue_order=['he2rna', 'vit'])

        # for slide in np.unique(dfs_[i]['slide'].values): 
        #     temp = dfs_[i]
        #     p = stats.ttest_rel(temp[(temp['study']=='vit') & (temp['slide']==slide)][metric].values,
        #                             temp[(temp['study']=='he2rna') & (temp['slide']==slide)][metric].values).pvalue

        #     height = ax.get_ylim()[1] * 0.95
        #     labels = [i.get_text() for i in ax.get_xticklabels()]
        #     j = int(np.where(np.asarray(labels)==slide)[0])
        #     #ax.hlines(y=height, xmin=j-0.1, xmax=j+0.1, linewidth=1, color='gray')
        #     ax.text(j, height, p_to_text(p), ha='center', va='bottom')

        plt.xlabel('slide')
        plt.ylabel(metric)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.savefig(f'{dest_folder0}/{metric}.png', dpi=300, bbox_inches='tight')
    
    import pdb; pdb.set_trace()

    # # plot comparison when taking best genes of model X (he2rna/vit) and compare the performance with performance of model Y on the those same genes
    # for s in all_studies:

    #     best_genes_of_study = s # 'he2rna' # study for which we chose the best 500 genes 

    #     top_K_genes_study = s # study for which we choose the top K spatial genes
    #     other_study = 'vit' if top_K_genes_study == 'he2rna' else 'he2rna'

    #     for slide_selection in [None, [i for i in slide_nrs if i not in remove_slides]]: #['269', '313', '334', '251', '243']

    #         dest_folder = f'results{suff_p}/{best_genes_of_study}_best500genes/{top_K_genes_study}_vs_{other_study}/'

    #         studies = ['vit', 'he2rna']
    #         prefix = 'vit_' if best_genes_of_study == 'vit' else 'he2rna_'
    #         folders = [f'{prefix}best_500genes_{stud}/' for stud in studies]
            
    #         total_dfs = []
    #         remove = []

    #         for i, folder in enumerate(folders):
    #             preds = {}
    #             all_dfs = []
                
    #             for slide_nr in slide_nrs:
    #                 slide_name = 'HRI_'+str(slide_nr)+'_T.tif'
    #                 csv_path = path + folder + slide_name + '/metrics.csv'

    #                 df = pd.read_csv(csv_path)
    #                 genes_rem = df.loc[df['nr_gt_vals'+suff]<nr_gt_vals_min].gene.values
    #                 for m in ['corr', 'sens', 'emd']:
    #                     remove += [[m+'_'+slide_nr, folder.split('_')[-1].replace('/', ''), gene] for gene in genes_rem]

    #                 all_dfs.append(df.rename(columns={'corr'+suff:'corr_'+slide_nr,
    #                                                   'sens'+suff:'sens_'+slide_nr, 
    #                                                   'emd'+suff:'emd_'+slide_nr})[['gene','corr_'+slide_nr,'sens_'+slide_nr,'emd_'+slide_nr]])
                
    #             tot_df = pd.concat(all_dfs, axis=1)

    #             total = tot_df.T.drop_duplicates()
    #             total.columns = total.loc['gene'].values
    #             total = total.loc[[i for i in total.index if i != 'gene']]
    #             total['study'] = studies[i]

    #             total_dfs.append(total)

    #         total_df = pd.concat(total_dfs)
    #         if len(remove) > 0:
    #             rem_df = pd.DataFrame(remove)
    #             rem_df.columns = ['index', 'study', 'variable']
            
    #         if proteincoding_only:
    #             genedf = pd.read_csv('/oak/stanford/groups/ogevaert/data/Gen-Pred/ref_files/ensembl_gene_map.csv')
    #             okgenes = genedf[genedf['biotype']=='protein_coding']['hgnc'].values
    #             total_df = total_df[[i for i in total_df.columns if i in okgenes] + ['study']]

    #         if slide_selection != None:
    #             total_df = total_df.loc[[ind for ind in total_df.index if ind.split('_')[1] in slide_selection]]
    #             dest_folder = dest_folder + '/' + '-'.join(slide_selection) + '/'
    #         else:
    #             dest_folder = dest_folder + '/' + 'all_slides' + '/'
    #         if not os.path.exists(path + dest_folder):
    #             os.makedirs(path + dest_folder)

    #         #################################### corrs, sens per slide
    #         for metric in ['corr', 'sens', 'emd']:
    #             total_part = total_df.loc[[col for col in total_df.index if metric in col]].drop_duplicates()

    #             fillval = 0 if ((metric == 'corr') or (metric == 'sens')) else 1e10
    #             total_part = total_part.fillna(fillval)

    #             if len(remove) > 0:
    #                 rem_df_ = rem_df[rem_df['index'].str.contains(metric)].drop_duplicates()

    #             if (metric == 'emd') and (norm_by != None):
    #                 cols = [col for col in total_part.columns if col != 'study']
    #                 for sid in total_part.index:
    #                     total_part.loc[sid,cols] = total_part.loc[sid][cols]/np.sqrt(area_info[area_info['id']==sid][norm_by].values[0])

    #             temp = total_part.reset_index()
    #             temp2 = pd.melt(temp, id_vars=['index','study'],value_vars=[col for col in temp.columns if col not in ['index','study']])

    #             if len(remove) > 0:
    #                 # remove genes with artefacts in gt
    #                 temp2 = pd.merge(temp2, rem_df_, on=['index','study','variable'], 
    #                                 how='outer', indicator=True).query("_merge != 'both'").drop('_merge', axis=1).reset_index(drop=True)

    #             plt.figure(figsize=(20,5))
    #             sns.violinplot(x=temp2['index'].values, y=temp2['value'].values.astype(np.float32), 
    #                                 hue=temp2['study'].values, palette=customPalette,
    #                                 hue_order=['he2rna', 'vit'])
    #             plt.savefig(path + dest_folder + f'{metric}_per_slide.png',bbox_inches='tight')

    #             ################################### get K best genes
    #             # Set the value of K
    #             K = [10, 20, 50]
    #             asc = False if ((metric == 'corr') or (metric == 'sens')) else True
    #             for k in K:
    #                 temp = total_part[total_part['study']==top_K_genes_study]
    #                 means = temp.median(axis=0)
    #                 sorted_means = means.sort_values(ascending=asc)
    #                 top_K_columns = sorted_means.head(k)
    #                 result = total_part[top_K_columns.index.tolist()+['study']]
    #                 result = result.reset_index()
    #                 result = pd.melt(result, id_vars=['index','study'],value_vars=[col for col in result.columns if col not in ['index','study']])

    #                 plt.clf()
    #                 plt.figure(figsize=(20,5))
    #                 sns.violinplot(x=result.variable, y=result['value'].astype(np.float32),
    #                                 hue=result.study, palette=customPalette, hue_order=['he2rna','vit'])
    #                 plt.xticks(rotation=90)
    #                 plt.savefig(path + dest_folder + f'{metric}_violin_top{k}_genes.png', bbox_inches='tight')


                    



