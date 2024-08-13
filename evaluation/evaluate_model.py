import pandas as pd
import numpy as np

import pickle as pl
import pdb
import os

from sklearn.metrics import mean_squared_error
from statsmodels.stats.multitest import fdrcorrection
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt

from evaluation.CorrelationStats import dependent_corr

if __name__=='__main__':
    
    model_dir = 'model_path'
    folds = 5
    cancers = ['brca', 'coad', 'gbm', 'kirp', 'kirc', 'luad', 'lusc', 'paad', 
              'prad', 'skcm', 'thca', 'ucec', 'hnsc', 'stad', 'blca', 'lihc']    

    save_path = os.path.join(model_dir, 'results')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df_list = []
    for cancer_type in cancers:
        try:
            print(cancer_type)
            with open(os.path.join(model_dir, cancer_type, 'test_results.pkl'), 'rb') as f:
                test_res = pl.load(f)

            real = []
            pred = []
            random = []
            wsi = []
            genes = test_res['genes']

            for k in range(folds):
                data = test_res[f'split_{k}']
                n_sample = len(data['preds'][:, 0])
                wsi.extend(data['wsi_file_name'])
                real.append(pd.DataFrame(data['real'], index = data['wsi_file_name'], columns = genes))
                pred.append(pd.DataFrame(data['preds'], index = data['wsi_file_name'], columns = genes))
                random.append(pd.DataFrame(data['random'], index = data['wsi_file_name'], columns = genes))    
                    
            df_real = pd.concat(real)
            df_pred = pd.concat(pred)
            df_random = pd.concat(random)

            #Make sure the index (samples) are identical in all the dataframes
            assert np.all(df_real.index == df_pred.index)
            assert np.all(df_real.index == df_random.index)

            pred_r = []
            random_r = []
            test_p = []
            pearson_p = []
            rmse_pred = []
            rmse_random = []
            rmse_quantile_norm = []
            rmse_mean_norm = []
            valid_genes = []

            for i, gene in enumerate(genes):
                real = df_real.loc[:, gene]
                pred = df_pred.loc[:, gene]
                random = df_random.loc[:, gene]
                
                if len(set(pred)) == 1 or len(set(real)) ==1 or len(set(random)) == 1:
                    xy, xy, yz = 0, 0, 0
                    p1, p2, p3, p = 1, 1, 1, 1
                else:
                    xy, p1 = stats.pearsonr(real, pred)
                    xz, p2 = stats.pearsonr(real, random)
                    yz, p3 = stats.pearsonr(pred, random)
                    t, p = dependent_corr(xy, xz, yz, len(real), twotailed=False, conf_level=0.95, method='steiger')

                pred_r.append(xy)
                random_r.append(xz)
                test_p.append(p)
                pearson_p.append(p1)

                # RMSE test
                rmse_p = mean_squared_error(real, pred, squared=False)
                rmse_r = mean_squared_error(real, random, squared=False)
                rmse_q = rmse_p / (np.quantile(real, 0.75) - np.quantile(real, 0.25) + 1e-5)
                rmse_m = rmse_p / np.mean(real)

                rmse_pred.append(rmse_p)
                rmse_random.append(rmse_r)
                rmse_quantile_norm.append(rmse_q)
                rmse_mean_norm.append(rmse_m)
                valid_genes.append(gene)
            
            combine_res = pd.DataFrame({'pred_real_r': pred_r,\
                                    'random_real_r': random_r,\
                                    'pearson_p': pearson_p,\
                                    'Steiger_p': test_p,\
                                    'rmse_pred': rmse_pred, \
                                    'rmse_random': rmse_random,
                                    'rmse_quantile_norm': rmse_quantile_norm,
                                    'rmse_mean_norm': rmse_mean_norm}, 
                                    index=valid_genes)

            combine_res = combine_res.sort_values('pred_real_r', ascending = False)

            # In case of constant values, replace correlation coefficient to 0
            combine_res['pred_real_r'] = combine_res['pred_real_r'].fillna(0)
            combine_res['random_real_r'] = combine_res['random_real_r'].fillna(0)

            # Correct pearson p values
            combine_res['pearson_p'] = combine_res['pearson_p'].fillna(1)
            _, fdr_pearson_p = fdrcorrection(combine_res['pearson_p'])
            combine_res['fdr_pearson_p'] = fdr_pearson_p
            
            # Correct steiger p values
            combine_res['Steiger_p'] = combine_res['Steiger_p'].fillna(1)
            _, fdr_Steiger_p = fdrcorrection(combine_res['Steiger_p'])
            combine_res['fdr_Steiger_p'] = fdr_Steiger_p

            combine_res['cancer'] = cancer_type
            df_list.append(combine_res)

        except:
            print(f'no data for {cancer_type}')

    all_res = pd.concat(df_list)
    sig_res = all_res[(all_res['pred_real_r'] > 0) & \
                    (all_res['pearson_p'] < 0.05) & \
                    (all_res['rmse_pred'] < all_res['rmse_random']) & \
                    (all_res['pred_real_r'] > all_res['random_real_r']) & \
                    (all_res['Steiger_p'] < 0.05) & \
                    (all_res['fdr_Steiger_p'] < 0.2)]

    all_res.to_csv(os.path.join(save_path, 'all_genes.csv'))
    sig_res.to_csv(os.path.join(save_path, 'sig_genes.csv')) 

    df_num_sig = sig_res['cancer'].value_counts().reset_index()
    df_num_sig.columns = ['cancer', 'num_genes']
    df_num_sig.to_csv(os.path.join(save_path, 'num_sign_genes.csv'))