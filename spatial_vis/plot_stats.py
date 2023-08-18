import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__=='__main__':
    path = '/oak/stanford/groups/ogevaert/data/Gen-Pred/visualizations/comparisons/'
    folder = 'vit_best_500genes_vit/'
    slide_nrs = ["242", "243", "248", "251", "255",
                 "259", "260", "262", "265", "266",
                 "268", "269", "270", "275", "296", 
                 "313", "334", "270", "304"] 
    preds = {}
    all_dfs = []
    
    for slide_nr in slide_nrs:
        slide_name = 'HRI_'+str(slide_nr)+'_T.tif'
        csv_path = path + folder + slide_name + '/corr_pval.csv'

        df = pd.read_csv(csv_path)
        all_dfs.append(df.rename(columns={'corr':'corr_'+slide_nr,'sens':'sens_'+slide_nr, 'emd':'emd_'+slide_nr})[['gene','corr_'+slide_nr,'sens_'+slide_nr,'emd_'+slide_nr]])
    
    tot_df = pd.concat(all_dfs,axis=1)
    tot_df = tot_df.loc[:,~tot_df.columns.duplicated()].copy()

    # plt.figure(figsize=(20,500))
    # cols=[col for col in tot_df.columns if col !='gene']
    # sns.heatmap(tot_df[cols],annot=True,cmap='coolwarm',yticklabels=tot_df['gene'].values,
    #             xticklabels=[i.split('_')[1] for i in cols])
    
    # plt.savefig(path + folder + 'corr_coeffs.png')

    total = tot_df.T
    total = total.fillna(0)
    total.columns = total.iloc[0].values
    total = total.iloc[1:]

    #################################### corrs, sens per slide

    for metric in ['corr', 'sens', 'emd']:
        total_part = total.loc[[col for col in total.index if metric in col]]
    
        plt.figure(figsize=(20,5))
        sns.violinplot(data=total_part.T)
        plt.savefig(path + folder + f'{metric}_per_slide.png',bbox_inches='tight')

        #################################### genes with corr > 0.1 -> how often does this occur across slides
        if metric == 'corr':
            l_ = (total_part>0.1).sum().values.tolist()
            counts = {}
            for i in range(9): 
                counts[i] = l_.count(i)
            x = list(counts.keys())
            y = list(counts.values())
            plt.bar(x, y)
            for i, v in enumerate(y): 
                plt.text(i, v, str(v), ha='center', va='bottom')
            plt.xticks(x)
            plt.xlabel('number of slides where correlation > 0.1')
            plt.ylabel('number of genes')
            plt.savefig(path + folder + f'{metric}_num_slides_genes.png')

        #################################### get K best genes
        # Set the value of K
        K = 50
        means = total_part.median(axis=0)
        sorted_means = means.sort_values(ascending=False)
        top_K_columns = sorted_means.head(K)
        result = total_part[top_K_columns.index]

        sns.set()
        sns.set_style("dark")

        plt.clf()
        plt.figure(figsize=(20,5))
        sns.violinplot(data=result)
        plt.xticks(rotation=90)
        plt.savefig(path + folder + f'{metric}_violin_top{K}_genes.png', bbox_inches='tight')

        



