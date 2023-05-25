import os.path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')


## Read RAP++_old
rap_df = pd.read_csv('rap++_ml_results/acs_NY/RAP(Marginal&Halfspace)/LR/result.csv')

rap_df = rap_df[['epsilon', 'seed', 'label', '(macro) f1']]
rap_df.rename(columns={'label': 'Target',
                       '(macro) f1': 'Score',
                       'seed': 'Seed',
                       'epsilon': 'Epsilon'},
              inplace=True)
# rap_df = rap_df[(rap_df['Target'] == 'PINCP')]

rap_df['Method'] = 'RAP++_old'
rap_df['Model'] = 'LR'
rap_df['Metric'] = 'F1'
rap_df['Is DP'] = 'Yes'
rap_df['Dataset'] = 'folktables_2014_multitask_NY'


privga_df = pd.read_csv('results/results_folktables_2014_multitask_NY_PrivGA.csv')

results = pd.read_csv('results.csv', index_col=None)

results = results[results['Metric'] == 'F1']
results = results[results['Epsilon'] <= 1]


results = pd.concat([results, rap_df,
                     privga_df])
results = results.sort_values(by='Epsilon')


results = results[results['Target'] == 'PINCP']
# results['Epsilon'] = results['Epsilon'].astype(str)
g = sns.relplot(data=results, x='Epsilon', y='Score', col='Model',
                hue='Method', row='Target', kind='line',
            style='Is DP', style_order=['Yes', 'No'], marker='o',
            facet_kws={'sharey': 'row'}, linewidth= 3.5)
g.fig.suptitle('ACS-Multitask-NY')
g.fig.subplots_adjust(top=.9)
g.fig.subplots_adjust(top=.9)

g.set(ylim=(0.5, 1))
plt.show()