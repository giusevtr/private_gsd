import pandas as pd


df = pd.read_csv('gsd_oneshot_cat.csv')

# df = df[df['error type'] == 'Max']
df = df[df['error type'] == 'Average']
df = df[(df['epsilon'] == 0.07) | (df['epsilon'] == 1.00)]
# df2 = df.groupby(['Data', 'epsilon', 'error type']).mean()[['time', 'error']]
df2 = df.groupby(['Data', 'epsilon', 'error type']).mean()[['error', 'time']]
print(df2)


# df_rp = pd.read_csv('rp_one_shot.csv')
# df_rp = df_rp.groupby(['dataset', 'epsilon']).mean()[['error_max', 'runtime']]
# print(df_rp)
