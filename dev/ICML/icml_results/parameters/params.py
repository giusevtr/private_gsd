
import pandas as pd


df = pd.read_csv('gsd_parameters.csv')


df = df[df['error type'] == 'Average']
df2 = df.groupby(['Mutations','Crossover']).mean()[['time', 'error']]

print(df2)
