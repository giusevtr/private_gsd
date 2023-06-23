import pandas as pd
from dev.generate_plots.plot_utils_results import read_result, show_result

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1.75)
sns.set_style("whitegrid")

gsd_df = pd.read_csv('icml_results/gsd_adaptive_prefix_temp.csv')
gsd_df['Data'] = gsd_df['Data'].apply(lambda x: x[16:-3])
cols0 = ['Generator','Data', 'Statistics', 'T', 'S', 'epsilon',  'error type']
gsd_df_2 = gsd_df.groupby(cols0)[['error']].mean().reset_index()
cols1 = ['Generator','Data', 'Statistics', 'epsilon',  'error type']
gsd_df_3 = gsd_df.groupby(cols1)[['error']].min().reset_index()
# gsd_df_3 = gsd_df.groupby(cols1, as_index=False).min()['error']
print(gsd_df_3)
def line_scatter_plot(x, y, **kwargs):
   print(kwargs)
   plt.plot(x, y, linewidth=3, **kwargs)
   plt.scatter(x, y, s=50, linewidth=4, **kwargs)


g = sns.FacetGrid(gsd_df_3, hue='Generator', col='Data', row='error type',
                  height=4,
                  aspect=2,
                  legend_out=False,
                  sharey=False
                  )
# g = sns.relplot(data=gsd_df_3, x='epsilon', y='error', hue='Generator',
#                 col='Data', row='error type', kind='line')

g.map(line_scatter_plot, "epsilon", "error")
g.add_legend()
fontsize = 20
for ax in g.axes.flat:
    # ax.set(s=40)
    epsilon_vals = [0.07, 0.23, 0.52, 0.74, 1.00]
    title = ax.get_title()
    stat_name = title.split(' ')[-1]
    error_type = title.split(' ')[3]
    print(error_type)
    if error_type == 'Average':
        ax.set_title(stat_name, fontsize=28)
    else:
        ax.set_title('')

    print(ax)
    ax.set_xticks(epsilon_vals, rotation=30)
    ax.set_xticklabels(rotation=0, labels=epsilon_vals)
    ax.set_xlabel(rf'$\epsilon$', fontsize=fontsize)
    ax.set_ylim([0, None])

    if error_type == 'Max':
        ax.set_ylabel(f'Max Error', fontsize=26)
    elif error_type == 'Average':
        ax.set_ylabel(f'Average Error', fontsize=26)

plt.subplots_adjust(top=0.94, bottom=0.18)
plt.show()
