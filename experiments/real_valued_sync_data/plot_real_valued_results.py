import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# dataname = 'folktables_2018_mobility_CA'

ERROR_TYPE = 'max error'
# ERROR_TYPE = 'l1 error'

##################################################
# Process PrivGA data
df = pd.read_csv('acs_numeric_results.csv', )
df = df[['data', 'generator','T', 'epsilon', 'seed',
       'max error', 'l1 error']]

df = df.groupby(['data', 'generator',  'T', 'epsilon'], as_index=False)[ERROR_TYPE].mean()
df = df.groupby(['data', 'generator',   'epsilon'], as_index=False)[ERROR_TYPE].min()
##################################################


g = sns.relplot(data=df, x='epsilon', y=ERROR_TYPE, hue='generator', col='data',   kind='line',
                marker='o',
                facet_kws={'sharey': False, 'sharex': True})

g.set(ylim=(0, None))

plt.subplots_adjust(top=0.9)
# plt.title(f'Real valued data with \nquery class range-marginals')
g.fig.suptitle(f'ACS-train real-valued with range queries.')

plt.show()