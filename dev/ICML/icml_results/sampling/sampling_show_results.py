import pandas as pd

module = 'Prefix'
df = pd.read_csv('../sampling.csv')
df = df[df['Statistics'] == module]


df_max = df[df['error type'] == 'Max']

df_max = df_max.groupby(['Data', 'samples', 'error type']).mean()[['error']]
print(df_max)


df_ave = df[df['error type'] == 'Average']
df_ave = df_ave.groupby(['Data', 'samples', 'error type']).mean()[['error']]
print(df_ave)


