import itertools

import jax.random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    SEED = 2

    #######
    ## Plot
    #######
    df_privga = pd.read_csv(f'res/privga_results_{SEED}.csv')
    df_rap = pd.read_csv(f'res/rap_results_{SEED}.csv')
    df = pd.concat([df_privga, df_rap])
    df_melt = pd.melt(df, var_name='error type', value_name='error', id_vars=['epoch', 'algo', 'rho', 'rounds'])
    sns.relplot(data=df_melt, x='epoch', y='error', hue='algo', col='error type', row='rho', style='rounds', kind='line',
                facet_kws={'sharey': False, 'sharex': True})
    plt.show()