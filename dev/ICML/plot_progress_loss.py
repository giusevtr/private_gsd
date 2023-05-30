import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df_0 = pd.read_csv('gsd_progress_folktables_2018_travel_CA_40_40.csv')
df_1 = pd.read_csv('gsd_progress_folktables_2018_travel_CA_80_0.csv')
df_0['Type'] = 'Mutation And Crossover'
df_1['Type'] = 'Mutation Only'


df = pd.concat([df_0, df_1], ignore_index=True)


sns.relplot(data=df, x='G', y='L2', hue='Type', kind='line')
plt.show()