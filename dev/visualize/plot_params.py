import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


df = pd.read_csv('results_parameters.csv')

# df.argmin()
idx = df[['max error']].idxmin()
best = df.loc[idx]

for r in best.iterrows():
    print(r)

df = df[df['init_sigma']==0.1]
df['init_sigma'] = df['init_sigma'].astype(str)
# ,init_sigma,sigma_limit,rows_pertubed,perturbation_sparsity,epsilon,seed,l1 error,max error,time
sns.relplot(data=df,
            row='sigma_limit',
            y='max error',
            hue='init_sigma',
            col='rows_pertubed',
            x='perturbation_sparsity',
            kind='line')
plt.show()