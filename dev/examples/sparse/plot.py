import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_sparse(data_array, alpha=0.5, s=0.1, title='', save_path=None):
    plt.figure(figsize=(5, 5))
    plt.title(title)
    plt.scatter(data_array[:, 0], data_array[:, 1], c=data_array[:, 2].astype(int), alpha=alpha, s=s)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    df_all = pd.read_csv('sparse_results.csv')
    df_all = df_all.drop('round init error', axis=1)
    # types = {'average error': float, 'max error': float, 'round max error': float,'algo': str}
    # df_all = pd.concat(RESSULTS, ignore_index=True).astype(types)
    # df_melt =
    df_melt = pd.melt(df_all, var_name='error type', value_name='error', id_vars=['epoch', 'algo', 'eps', 'seed'])

    sns.relplot(data=df_melt, x='epoch', y='error', hue='algo', col='error type', row='eps', kind='line',
                facet_kws={'sharey': False, 'sharex': True})
    plt.show()