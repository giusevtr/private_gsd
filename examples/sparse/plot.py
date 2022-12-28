import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_all = pd.read_csv('sparse_results.csv')
df_all = df_all.drop('round init error', axis=1)
# types = {'average error': float, 'max error': float, 'round max error': float,'algo': str}
# df_all = pd.concat(RESSULTS, ignore_index=True).astype(types)
# df_melt =
df_melt = pd.melt(df_all, var_name='error type', value_name='error', id_vars=['epoch', 'algo', 'eps', 'seed'])

sns.relplot(data=df_melt, x='epoch', y='error', hue='algo', col='error type', row='eps', kind='line',
            facet_kws={'sharey': False, 'sharex': True})
plt.show()