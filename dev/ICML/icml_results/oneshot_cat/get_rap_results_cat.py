import pandas as pd





df = pd.read_csv('rp_one_shot.csv')
df = df[(df['epsilon'] == 0.07) | (df['epsilon'] == 1.00)]

df_rp = df.groupby(['dataset', 'epsilon']).mean()[['error_mean', 'runtime']]
print(df_rp)
