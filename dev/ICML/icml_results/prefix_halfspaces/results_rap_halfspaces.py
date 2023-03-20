import pandas as pd


print('RAP(Halfspaces)')
df = pd.read_csv('rap_halfspaces.csv')

df = df[df['error type'] == 'Average']
# df = df[df['error type'] == 'Max']
df2 = df.groupby(['Data', 'epsilon', 'error type']).mean()[['error']]
print(df2)

