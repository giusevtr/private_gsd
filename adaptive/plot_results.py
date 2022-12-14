import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

algorithms = ['RP++',
              'SimpleGA(popsize=100, topk=30, reg=False)',
              'SimpleGA(popsize=100, topk=30, reg=True)']

df_all = pd.concat([pd.read_csv(f'results/{algo}.csv') for algo in algorithms])

# df = pd.read_csv('results/results.csv')

df_melt = pd.melt(df_all, var_name='error type', value_name='error', id_vars=['epoch', 'algo'])
sns.relplot(data=df_melt, x='epoch', y='error', hue='algo',col='error type', kind='line', facet_kws={'sharey': False, 'sharex': True})
plt.show()
# plt.show()
# sns.relplot(data=df, x='epoch', y='max', hue='algo', kind='line')
