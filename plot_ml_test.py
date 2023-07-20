import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
sns.set_style("whitegrid")
sns.set(font_scale=1.2)
# sns.set_context("poster", font_scale = 1.0,
#                 rc={"grid.linewidth": 0.6})


df_sklearn = pd.read_csv('acs_sync_ml_results.csv')
df_catboost = pd.read_csv('results/acs_sync_catboost_results.csv')
df_catboost_orig = pd.read_csv('results/acs_original_catboost_results.csv')
df_original = pd.read_csv('results/acs_sync_ml_results_original.csv')
df = pd.concat((df_sklearn, df_catboost, df_original, df_catboost_orig), ignore_index=True)
# df['State'] = df['Public Dataset'].apply(lambda s : s[-2: ])
df['Data'] = df['Data'].apply(lambda s: s[16:-3])
df = df[df['Metric'] == 'accuracy']
df = df[df['Categorical Only'] == True]

df_sync = df[df['Type'] == 'Sync']
df_sync['N'] = df_sync['N'].astype(int)
df_r = df[df['Type'] == 'Original']

g = sns.relplot(data=df_sync, x='N', y='Score',
            row='Data',
            col='Model',
            col_order=['LogisticRegression', 'XGBoost', 'LightGBM', 'Catboost'], # hue='State',
            style='Eval Data',
            kind='line',
            linewidth=5,
            facet_kws={'sharey': 'row'})
axes = g.axes.flatten()
g.set(xscale="log")
for ax in axes:
    t = ax.get_title().split(' ')
    data = t[2]
    model = t[6]
    df_temp = df_r[(df_r['Data'] == data) & (df_r['Model'] == 'Constant')]
    v = df_temp['Score'].values[0]
    ax.hlines(y=v, xmin=0, xmax=32000, colors='k', linestyles='dashed')


    df_model = df_r[(df_r['Data'] == data) & (df_r['Model'] == model)]
    score_original = np.mean(df_model['Score'].values)
    ax.hlines(y=score_original, xmin=0, xmax=32000, colors='r', linestyles='dashed')


plt.show()