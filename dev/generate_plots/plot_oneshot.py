import numpy as np
import pandas as pd
# from generate_plots.plot_utils_results import read_result, show_result
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1.)
sns.set_style("whitegrid")


error_lbl = 'error_max'

def show_oneshot_result(df):
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
        df['Temp'] = 's'
        sns.lineplot(data=df, x='x', y='y', style='Temp', markers=['o'],  **kwargs)

    g = sns.FacetGrid(df,  hue='generator',  height=4,
                      legend_out=False,
                      row='error type',
                      col='data',
                      sharey=False,
                      aspect=1.0,
                      )
    g.map(line_scatter_plot, "epsilon", 'error', linewidth=3)
    g.add_legend()
    g.set(ylim=(0, None))
    plt.subplots_adjust(top=0.94, bottom=0.22, left=0.07)

    sns.move_legend(g, "lower center", bbox_to_anchor=(.5, -.02),
                    ncol=4, title=None, frameon=False, fontsize=34)

    for ax in g.axes.flat:
        # ax.set(s=40)
        epsilon_vals = [0.07, 0.23, 0.52, 0.74, 1.00]
        title = ax.get_title()
        new_title = title.split(' ')[-1]
        new_title = new_title.split('_')
        new_title = new_title[2].capitalize()

        error_type = title.split(' ')[3]
        # print(error_type)
        if error_type == 'error_max':
            ax.set_title(new_title, fontsize=34)
        else:
            ax.set_title('')
        #
        # print(ax)
        ax.set_xticks( epsilon_vals)
        ax.set_xticklabels( rotation=0, labels=epsilon_vals)
        ax.set_xlabel(rf'$\epsilon$', fontsize=fontsize)
        #
        if new_title == 'Coverage' :
            if error_type == 'max':
                ax.set_ylabel(f'Max Error', fontsize=34)
            elif error_type == 'l1':
                ax.set_ylabel(f'Average Error', fontsize=34)
        # ax.tick_params(axis='y', which='major', labelsize=16, rotation=45)
        ax.tick_params(axis='x', which='major', labelsize=16, rotation=45)
    # plt.title(f'Input data is {dataname} and \nquery class is categorical-marginals')
    plt.show()

# priv_ga_df = pd.read_csv('../results/cat_ranges_results/privga/result_cat_privga.csv', index_col=0)
# priv_ga_df = priv_ga_df.melt(id_vars=['data', 'generator', 'stats', 'T', 'epsilon', 'seed'], var_name='error type', value_name='error')
priv_ga_df = pd.read_csv('../results/privga_evaluate_3way_results.csv')
priv_ga_df = priv_ga_df.melt(id_vars=['data_name', 'generator', 'epsilon', 'seed', 'subgroup'], var_name='error type', value_name='error')
priv_ga_df = priv_ga_df.rename(columns={'data_name': 'data'})
# priv_ga_df['generator'] = 'PrivGA'
# df = pd.concat(res_df, ignore_index=True)
# dataname = 'folktables_2018_mobility_CA'
# df = df.loc[df['data'] == dataname, :]

show_oneshot_result(priv_ga_df)

