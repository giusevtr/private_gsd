
import matplotlib.pyplot as plt
import pandas as pd
from dp_data import load_domain_config, load_df, get_evaluate_ml
import numpy as np

import seaborn as sns
if __name__ == '__main__':
    # results = pd.read_csv('acsreal_results_Halfspaces.csv', index_col=0)
    results_privga = pd.read_csv('acsreal_results_PrivGA_Halfspaces.csv', index_col=0)
    results_privga['Method'] = 'PrivateGSD(HS)'

    results_privga_pr = pd.read_csv('acsreal_results_PrivGA_Prefix.csv', index_col=0)
    results_privga_pr['Method'] = 'PrivateGSD(Prefix)'

    results_rappp = pd.read_csv('acsreal_results_RAP++_Halfspaces.csv', index_col=0)
    results_rappp['Method'] = 'RAP++(HS)'

    # results_rappp_pr = pd.read_csv('acsreal_results_RAP++_Prefix.csv', index_col=0)
    # results_rappp_pr['Method'] = 'RAP++(Prefix)'

    results_rap = pd.read_csv('acsreal_results_RAP_Ranges.csv', index_col=0)
    results_rap['Method'] = 'RAP(Ranges)'


    results = pd.concat([results_rappp, results_privga, results_privga_pr, results_rap])

    df = results[['Method', 'target',  'Metric', 'Score', 'Epoch', 'Epsilon', 'Seed']]


    df = df.groupby(['Method', 'target',  'Metric',  'Epoch', 'Epsilon'], as_index=False)['Score'].mean()
    df = df.groupby(['Method', 'target',  'Metric',  'Epsilon'], as_index=False)['Score'].max()


    sns.relplot(data=df, x='Epsilon', y='Score', col='target', hue='Method', row='Metric', kind='line', facet_kws={'sharey':False})

    plt.show()

