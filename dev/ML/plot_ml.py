import os.path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

results = pd.read_csv('results.csv', index_col=None)

results['Epsilon'] = results['Epsilon'].astype(str)
# plt.title('ACS-Multitask-NY(Clipped)')
g = sns.relplot(data=results, x='Epsilon', y='Score', col='Target', hue='Method', row='Metric', kind='line',
            style='Is DP', style_order=['Yes', 'No'], marker='o',
            facet_kws={'sharey': False}, linewidth = 3.5)
g.fig.suptitle('ACS-Multitask-NY')
g.fig.subplots_adjust(top=.9)
plt.show()