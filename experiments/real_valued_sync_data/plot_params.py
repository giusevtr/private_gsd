import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# dataname = 'folktables_2018_mobility_CA'

ERROR_TYPE = 'max error'
# ERROR_TYPE = 'l1 error'

##################################################
# Process PrivGA data
df0 = pd.read_csv('param_results/folktables_2018_income_CA/PrivGA/privga_params_0.csv', )
df1 = pd.read_csv('param_results/folktables_2018_income_CA/PrivGA/privga_params_1.csv', )
df2 = pd.read_csv('param_results/folktables_2018_income_CA/PrivGA/privga_params_2.csv', )


df = pd.concat([df0, df1, df2])
df['pop'] = df['pop'].astype(str)
# df['mut'] = 10 * df['mut']
df['mut'] = df['mut'].astype(str)
# df['mut'] = 10 * df['mut']
# df['pop'] = 10 * df['mut']
df['data_size'] = df['data_size'].astype(str)
#################################################

df = df.groupby([ 'mate', 'elite', 'mut', 'data_size', 'pop', 'eps'], as_index=False)[['max_error', 'time']].mean()


# eps = 0.01
eps = 1
df = df[df['eps'] == eps]

print(df)

g = sns.relplot(data=df, x='time', y='max_error', hue='pop', style='data_size',
                size='mut',
                sizes={'1': 50, '5': 100},
            #     col='data_size',
            # row='pop',
                kind='scatter', facet_kws={'sharey': True, 'sharex': True})
plt.subplots_adjust(top=0.9)
g.fig.suptitle(f'epsilon={eps}')
g.set(ylim=(0, None))
plt.show()