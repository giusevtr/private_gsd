import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1.75)
sns.set_style("whitegrid")

# def read_result(path, error_type='max error'):
def read_result(data_path, error_type='max error'):
    # data_path = f'{data_dir}/mix/privga/result_mix_privga.csv'
    df = pd.read_csv(data_path)
    df = df.rename(columns={"dataset": "data",
                            "error_max": "max error", 'error_mean': 'l1 error',
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
    gens = ['PrivGA', 'GEM', 'RAP', 'PGM', 'RAP++']

    gens_in_df = df['generator'].unique()
    hue_order = []
    for g in gens:
        if g in gens_in_df:
            hue_order.append(g)

    print(hue_order)
    g = sns.relplot(data=df, x='epsilon', y=error_type,
                    # col='data',
                    hue='generator', kind='line',
                    marker='o',
                    # s=100,
                    facet_kws={'sharey': False, 'sharex': True, 'legend_out': False},
                    hue_order=hue_order,
                    col='Statistics',
                    aspect=1.5,
                    linewidth=5,
                    alpha=0.9)
    plt.subplots_adjust(top=0.9, bottom=0.26)

    sns.move_legend(g, "lower center", bbox_to_anchor=(.5, .005),
                    ncol=4, title=None, frameon=False)

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
