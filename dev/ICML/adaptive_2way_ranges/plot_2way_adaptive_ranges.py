import numpy as np
import pandas as pd
# from generate_plots.plot_utils_results import read_result, show_result
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1.)
sns.set_style("whitegrid")

Cols = ['Generator','Data','Statistics','epsilon','seed', 'error type', 'error']

df_gsd = pd.read_csv('icml_results/gsd_ranges.csv')


# df_rp = pd.read_csv('rp_one_shot.csv')


# df_rp['Generator'] = 'RAP'
# df_rp['Statistics'] = 'Range'
# df_rp.rename(columns={'dataset': 'Data', 'runtime': 'time', 'test_seed': 'seed', 'error_max': 'Max',
#                       'error_mean': 'Average'}, inplace=True)
#
# df_rp = df_rp.melt(id_vars=Cols[:-2], value_vars=['Max', 'Average'], var_name='error type', value_name='error')






df_gsd = df_gsd[Cols]
# df_rp = df_rp[Cols]
df = pd.concat([
    df_gsd,
                # df_rp
])


fontsize = 24
gens = ['GSD']

gens_in_df = df['Generator'].unique()
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

g = sns.FacetGrid(df,  hue='Generator',  height=4,
                  legend_out=False,
                  row='error type',
                  col='Data',
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
    if error_type == 'Max':
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
        if error_type == 'Max':
            ax.set_ylabel(f'Max Error', fontsize=34)
        elif error_type == 'Average':
            ax.set_ylabel(f'Average Error', fontsize=34)
    # ax.tick_params(axis='y', which='major', labelsize=16, rotation=45)
    ax.tick_params(axis='x', which='major', labelsize=16, rotation=45)
# plt.title(f'Input data is {dataname} and \nquery class is categorical-marginals')
plt.show()

