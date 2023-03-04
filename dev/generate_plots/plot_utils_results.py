import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1.75)
sns.set_style("whitegrid")

error_lbl = 'error'
# def read_result(path, error_type='max error'):
def read_result(data_path, error_type='max error'):
    # data_path = f'{data_dir}/mix/privga/result_mix_privga.csv'
    df = pd.read_csv(data_path)
    df = df.rename(columns={"dataset": "data",
                            "error_max": "max error", 'error_mean': 'l1 error',
                            "test_seed": "seed"})
    df[error_lbl] = df[error_type]
    df['error type'] = error_type
    replace_names = ['-train', '-cat-train']
    for rn in replace_names:
        df['data'] = df['data'].replace(to_replace=f"folktables_2018_mobility_CA{rn}",
                                        value="folktables_2018_mobility_CA")

    df = df[['data',  'T', 'epsilon', 'seed', 'error type', error_lbl]]

    df = df.groupby(['data', 'T', 'epsilon', 'error type'], as_index=False)[error_lbl].mean()
    df = df.groupby(['data', 'epsilon', 'error type'], as_index=False)[error_lbl].min()
    # privga_df['generator'] = 'PrivateGSD'
    return df


def show_result(df):
    fontsize = 24
    gens = ['PrivateGSD', 'GEM', 'RAP', 'PGM', 'RAP++']

    gens_in_df = df['generator'].unique()
    hue_order = []
    for g in gens:
        if g in gens_in_df:
            hue_order.append(g)

    print(hue_order)
    # g = sns.relplot(data=df,
    #                 x='epsilon',
    #                 y=error_type,
    #                 hue='generator', kind='line',
    #                 facet_kws={'sharey': False, 'sharex': True, 'legend_out': False},
    #                 hue_order=hue_order,
    #                 col='Statistics',
    #                 aspect=1.5,
    #                 # linewidth=5,
    #                 markers=True,
    #                 style='generator',
    #
    #                 marker='o',
    #                 scatter_kws={'s': 100},
    #                 # s=100,
    #                 alpha=0.9)

    def line_scatter_plot(x, y, **kwargs):
        print(kwargs)
        plt.plot(x, y, linewidth=3, **kwargs)
        plt.scatter(x, y, s=50, linewidth=4, **kwargs)

    g = sns.FacetGrid(df,  hue='generator',  col='Statistics', row='error type', height=4,
                      legend_out=False,
                      sharey=False, aspect=1.5
                      )
    g.map(line_scatter_plot, "epsilon", error_lbl)
    g.add_legend()
    plt.subplots_adjust(top=0.94, bottom=0.18)

    sns.move_legend(g, "lower center", bbox_to_anchor=(.5, -.02),
                    ncol=4, title=None, frameon=False, fontsize=20)

    for ax in g.axes.flat:
        # ax.set(s=40)
        epsilon_vals = [0.07, 0.23, 0.52, 0.74, 1.00]
        title = ax.get_title()
        stat_name = title.split(' ')[-1]
        error_type = title.split(' ')[3]
        print(error_type)
        if error_type == 'max':
            ax.set_title(stat_name, fontsize=28)
        else:
            ax.set_title('')

        print(ax)
        ax.set_xticks( epsilon_vals, rotation=30)
        ax.set_xticklabels( rotation=0, labels=epsilon_vals)
        ax.set_xlabel(rf'$\epsilon$', fontsize=fontsize)

        if stat_name == 'Halfspaces' or stat_name == 'Marginals':
            if error_type == 'max':
                ax.set_ylabel(f'Max Error', fontsize=26)
            elif error_type == 'l1':
                ax.set_ylabel(f'Average Error', fontsize=22)
        ax.tick_params(axis='y', which='major', labelsize=16, rotation=45)
        ax.tick_params(axis='x', which='major', labelsize=16, rotation=45)

    # plt.title(f'Input data is {dataname} and \nquery class is categorical-marginals')
    plt.show()
