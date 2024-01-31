# Assess the prediction for each individual gene across cancer types
import pandas as pd
import pickle as pl
import pdb
import os
from sklearn.metrics import mean_squared_error
from statsmodels.stats.multitest import fdrcorrection
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from CorrelationStats import dependent_corr
import sys

def extract_gene_names(df):
    genes = [g for g in df.columns if g.startswith("rna_")]
    genes = [g[4:] for g in genes]
    return genes 

model = "sequoia"
data_path = "./output"
save_path = "./output"

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Load output from trained results
cancers = ["brca", "coad", "gbm", "kirp", "kirc", "luad", "lusc", "paad", "prad"]
df_list = []

for cancer_type in cancers:
    folds = 5
    with open(os.path.join(data_path, cancer_type, "test_results.pkl"), 'rb') as f:
        test_res = pl.load(f)
    
    real = []
    pred = []
    random = []
    wsi = []

    for k in range(folds):
        data = test_res[f'split_{k}']
        n_sample = len(data['preds'][:, 0])
        print(f"fold {k}: n = {n_sample}")

        if model == "sequoia":
            real.append(pd.DataFrame(data['real']))
            pred.append(pd.DataFrame(data['preds']))
            random.append(pd.DataFrame(data['random']))
            genes = [g[4:] for g in data['genes']]
            wsi.extend(data['wsi_file_name'])
        else:
            real.append(pd.DataFrame(data['labels']))
            pred.append(pd.DataFrame(data['preds']))
            random.append(pd.DataFrame(data['preds_random']))
            genes = [g[4:] for g in data['genes']]

    df_real = pd.concat(real)
    df_pred = pd.concat(pred)
    df_random = pd.concat(random)

    pred_r = []
    random_r = []
    test_p = []
    pearson_p = []
    rmse_pred= []
    rmse_random = []
    valid_genes = []

    for i, gene in enumerate(genes):
        # Correlation test
        xy, p1 = stats.pearsonr(df_real.iloc[:, i], df_pred.iloc[:, i])
        xz, p2 = stats.pearsonr(df_real.iloc[:, i], df_random.iloc[:, i])
        yz, p3 = stats.pearsonr(df_pred.iloc[:, i], df_random.iloc[:, i])
        n = len(df_real.iloc[:, i])
        t, p = dependent_corr(xy, xz, yz, n, twotailed=False, conf_level=0.95, method='steiger')

        if None in (p1, p2, p3, p):
            continue

        pred_r.append(xy)
        random_r.append(xz)
        test_p.append(p)
        pearson_p.append(p1)

        # RMSE test
        rmse1 = mean_squared_error(df_real.iloc[:, i], df_pred.iloc[:, i],squared=False)
        rmse2 = mean_squared_error(df_real.iloc[:, i], df_random.iloc[:, i],squared=False)
        rmse_pred.append(rmse1)
        rmse_random.append(rmse2)
        valid_genes.append(gene)
    
    combine_res = pd.DataFrame({"pred_real_r": pred_r,\
                            "random_real_r": random_r,\
                            'pearson_p': pearson_p,\
                            "Steiger_p": test_p,\
                            'rmse_pred': rmse_pred, \
                            'rmse_random': rmse_random}, index=valid_genes)

    combine_res = combine_res.sort_values('pred_real_r', ascending = False)
    combine_res = combine_res[~combine_res['Steiger_p'].isna()]
    _, fdr_p = fdrcorrection(combine_res['Steiger_p'])
    combine_res['fdr_Steiger_p'] = fdr_p
        
    sig_res = combine_res[(combine_res['pred_real_r'] > 0) & \
                    (combine_res['pearson_p'] < 0.05) & \
                    (combine_res['pred_real_r'] > combine_res['random_real_r']) & \
                    (combine_res['Steiger_p'] < 0.05) & \
                    (combine_res['fdr_Steiger_p'] < 0.2)]
    
    sig_res['cancer'] = cancer_type
    print(f"Found {sig_res.shape[0]} significant genes")

    if sig_res.shape[0] > 0:
        df_list.append(sig_res)
        
df_final = pd.concat(df_list)
df_final.to_csv(os.path.join(save_path, "sig_genes.csv"))

df_num_sig = df_final['cancer'].value_counts().reset_index()
df_num_sig.columns = ['cancer', 'num_genes']
df_num_sig.to_csv(os.path.join(save_path, "num_sign_genes.csv"))