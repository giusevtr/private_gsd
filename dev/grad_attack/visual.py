import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_real = pd.read_csv('real_train.csv')
df_real['Type'] = 'Target'
df_sync = pd.read_csv('sync_data_50_1000.csv')
df_sync['Type'] = 'Reconstructed'


df_all = pd.concat([df_real, df_sync])

sns.relplot(data=df_all, x='f0', y='f1',hue='label', col='Type')
plt.show()




