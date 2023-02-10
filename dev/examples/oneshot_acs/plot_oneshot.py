import numpy as np
import pandas as pd
# from generate_plots.plot_utils_results import read_result, show_result
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1.)
sns.set_style("whitegrid")


error_lbl = 'error_max'

def show_oneshot_result(df):
    # df = df.melt(id_vars=['generator', 'epsilon','seed'])


    fontsize = 24
    gens = ['PrivGA',  'RAP']

    gens_in_df = df['generator'].unique()
    hue_order = []
    for g in gens:
        if g in gens_in_df:
            hue_order.append(g)


    def line_scatter_plot(x, y, **kwargs):
        print(kwargs)
        # plt.plot(x, y, linewidth=1, **kwargs)
        # plt.scatter(x, y, s=10, linewidth=2, **kwargs)
        df = pd.DataFrame(np.column_stack((x, y)), columns=['x', 'y'])
        sns.lineplot(data=df, x='x', y='y', **kwargs)

    g = sns.FacetGrid(df,
                      hue='generator',
                      # height=4,
                      # legend_out=False,
                      sharey=False, aspect=1.5,
                      )
    g.map(line_scatter_plot, "epsilon", error_lbl)
    g.add_legend()
    plt.subplots_adjust(top=0.94, bottom=0.28)

    # sns.move_legend(g, "lower center", bbox_to_anchor=(.5, -.02),
    #                 ncol=4, title=None, frameon=False, fontsize=20)

    for ax in g.axes.flat:
        # ax.set(s=40)
        epsilon_vals = [0.07, 0.23, 0.52, 0.74, 1.00]
        # title = ax.get_title()
        # stat_name = title.split(' ')[-1]
        # error_type = title.split(' ')[3]
        # print(error_type)
        # if error_type == 'max':
        #     ax.set_title(stat_name, fontsize=28)
        # else:
        #     ax.set_title('')
        #
        # print(ax)
        # ax.set_xticks( epsilon_vals, rotation=30)
        # ax.set_xticklabels( rotation=0, labels=epsilon_vals)
        # ax.set_xlabel(rf'$\epsilon$', fontsize=fontsize)
        #
        # if stat_name == 'Halfspaces' or stat_name == 'Marginals':
        #     if error_type == 'max':
        #         ax.set_ylabel(f'Max Error', fontsize=26)
        #     elif error_type == 'l1':
        #         ax.set_ylabel(f'Average Error', fontsize=22)
        # ax.tick_params(axis='y', which='major', labelsize=16, rotation=45)
        # ax.tick_params(axis='x', which='major', labelsize=16, rotation=45)

    # plt.title(f'Input data is {dataname} and \nquery class is categorical-marginals')
    plt.show()

# priv_ga = pd.read_csv('../ICML/one_shot/privga_oneshot.csv')
priv_ga = pd.read_csv('../examples/oneshot_acs/privga_2way_results.csv')
priv_ga['generator'] = 'PrivGA'
# df = pd.concat(res_df, ignore_index=True)
# dataname = 'folktables_2018_mobility_CA'
# df = df.loc[df['data'] == dataname, :]

show_oneshot_result(priv_ga)

