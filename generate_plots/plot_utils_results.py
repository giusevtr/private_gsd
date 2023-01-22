import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")
sns.set(font_scale=1.5)

def read_result(path, error_type='max error'):
    df = pd.read_csv(path)
    df = df.rename(columns={"dataset": "data", "error_max": "max error", 'error_mean': 'l1 error',
                            "test_seed": "seed"})

    replace_names = ['-train', '-cat-train']
    for rn in replace_names:
        df['data'] = df['data'].replace(to_replace=f"folktables_2018_mobility_CA{rn}",
                                        value="folktables_2018_mobility_CA")

    df = df[['data',  'T', 'epsilon', 'seed', 'max error', 'l1 error']]

    df = df.groupby(['data', 'T', 'epsilon'], as_index=False)[error_type].mean()
    df = df.groupby(['data', 'epsilon'], as_index=False)[error_type].min()
    # privga_df['generator'] = 'PrivGA'
    return df


def show_result(df, error_type='max error'):
    fontsize = 18
    g = sns.relplot(data=df, x='epsilon', y=error_type,
                    col='data',
                    hue='generator', kind='line',
                    marker='o',
                    facet_kws={'sharey': False, 'sharex': True},
                    hue_order=['PrivGA', 'GEM', 'RAP', 'PGM'],
                    aspect=1.5,
                    linewidth=3,
                    alpha=0.9)
    plt.subplots_adjust(top=0.9)

    for ax in g.axes.flat:
        epsilon_vals = [0.07, 0.23, 0.52, 0.74, 1.00]
        ax.set_xticks(epsilon_vals)
        ax.set_xticklabels( rotation=0, labels=epsilon_vals)
        ax.set_xlabel(f'epsilon', fontsize=fontsize)
        if error_type == 'max error':
            ax.set_ylabel(f'Max Error', fontsize=fontsize)
        elif error_type == 'l1 error':
            ax.set_ylabel(f'Average Error', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # plt.title(f'Input data is {dataname} and \nquery class is categorical-marginals')
    plt.show()
