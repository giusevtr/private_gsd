import os.path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')

results = pd.read_csv('results.csv', index_col=None)

results = results[results['Metric'] == 'F1']

results['Epsilon'] = results['Epsilon'].astype(str)
# plt.title('ACS-Multitask-NY(Clipped)')
g = sns.relplot(data=results, x='Epsilon', y='Score', col='Model', hue='Method', row='Target', kind='line',
            style='Is DP', style_order=['Yes', 'No'], marker='o',
            facet_kws={'sharey': 'row'}, linewidth= 3.5)
g.fig.suptitle('ACS-Multitask-NY')
g.fig.subplots_adjust(top=.9)
plt.show()