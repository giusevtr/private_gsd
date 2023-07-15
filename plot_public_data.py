import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



df = pd.read_csv('public_data_results_cat.csv', index_col=0)
df['State'] = df['Public Dataset'].apply(lambda s : s[-2: ])

df = df.melt(
    id_vars=['eps', 'Private Dataset', 'State', 'Model'],
    value_vars=['Sync f1', 'Real f1', 'Public f1'],
            var_name='Type',
             value_name='F1 score')
sns.relplot(data=df, x='eps', y='F1 score',
            row='Private Dataset',
            col='Model',
            hue='State',
            style='Type',
            kind='line',
            linewidth=5,
            facet_kws={'sharey': False})
plt.show()