import pandas as pd
import seaborn as sns
sns.set(font_scale=1.75)
sns.set_style("whitegrid")

import matplotlib.pyplot as plt


clf_error_label = 'Classification Error'
target_label = 'Target'
# def read_result(path, error_type='max error'):
def read_result(data_path, ml_model):

    # data_path = f'{data_dir}/mix/privga/result_mix_privga.csv'
    df = pd.read_csv(data_path, index_col=0)
    df = df.rename(columns={"algo":"generator"})
    df['generator'] = df['generator'].replace("PrivGA(Halfspaces)", "PrivGA(HS)")
    df['model'] = ml_model
    df[target_label] = df['error type'].apply(lambda s: s.split()[0])
    # df[clf_error_label] = 1 - df['private accuracy']

    targets = df[target_label].unique()
    eps_values = df['epsilon'].unique()
    orig_rows = []
    for t in targets:
        temp_df = df.loc[df[target_label] == t, :]
        target_acc = temp_df['original accuracy'].values[0]
        for eps in eps_values:
            orig_rows.append(['Original', eps, t, ml_model, 1-target_acc])
    orig_df = pd.DataFrame(orig_rows, columns=['generator', 'epsilon', target_label, 'model', clf_error_label])

    df = df[[ 'generator', 'T', 'epsilon', 'seed', target_label, 'model', 'original accuracy', 'private accuracy']]
    df[clf_error_label] = 1 - df['private accuracy']

    # df.melt(id_vars=['data',  'algo', 'T', 'epsilon', 'seed'], )
    df = df.groupby(['generator', 'T', 'epsilon', target_label, 'model'], as_index=False)[clf_error_label].mean()
    df = df.groupby(['generator', 'epsilon', target_label, 'model'], as_index=False)[clf_error_label].min()
    # privga_df['generator'] = 'PrivGA'


    final_df = pd.concat([df, orig_df], ignore_index=True)

    # final_df = final_df.set_index('generator')
    # # reorder the index with the order given in list 'months_ordered'
    # months_ordered = ['Original', 'PrivGA(HS)', 'PrivGA(Prefix)', 'RAP(Ranges)', 'GEM(Range)']
    # final_df = final_df.reindex(months_ordered)


    return final_df


def show_result(df):

    gen = ['Original', 'PrivGA(HS)', 'PrivGA(Prefix)', 'RAP(Ranges)', 'GEM(Range)']
    # gen_bold = ['Original', r'\textbf{PrivGA(HS)}', 'PrivGA(Prefix)', 'RAP(Ranges)', 'GEM(Range)']

    # sns.barplot(data=df, x='generator', y='ML accuracy', hue='ML method type', row='epsilon')
    g = sns.relplot(data=df,
                    # x="generator",
                    x=clf_error_label,
                    y='generator',
                    # kind="bar",
                    # row='epsilon',
                    col=target_label,
                    # col='target',
                    style='model',
                    hue='model',
                    # mar
                    # hue_order=['PrivGA(Halfspaces)', 'PrivGA(Prefix)', 'RAP(Ranges)', 'GEM(Ranges)',  'Original'],
                    facet_kws={'sharey': True, 'sharex': False, 'legend_out': False},
                    # sharey=False,
                    s=200,
                    # order=gen
                    # order=gen
                )
    # g.set_xticklabels( rotation=0)
    # g.despine(left=True)
    # g.add_legend([])
    sns.move_legend(g, "lower center", bbox_to_anchor=(.5, .85),
                    ncol=5, title=None, frameon=False, fontsize=28)

    # plt.legend(loc='bottom', title='Team')
    plt.subplots_adjust(top=0.8, bottom=0.15)

    g.set_xlabels(clf_error_label)
    for ax in g.axes.flat:
        title = ax.get_title()
        ax.set_title(title, fontsize=28)
        temp = ax.get_xticklabels()
        print(temp)
        # ax.set(yticklabels=[])
        # ax.set_yticklabels( rotation=0, labels=['', '', '', '', ''])
        # ax.set_xlabel('')
        ax.set_ylabel('')
        # ValueError: keyword fontsize is not recognized; valid keywords are ['size', 'width', 'color', 'tickdir', 'pad', 'labelsize', 'labelcolor', 'zorder', 'gridOn', 'tick1On', 'tick2On', 'label1On', 'label2On', 'length', 'direction', 'left', 'bottom', 'right', 'top', 'labelleft', 'labelbottom', 'labelright', 'labeltop', 'labelrotation', 'grid_agg_filter', 'grid_alpha', 'grid_animated', 'grid_antialiased', 'grid_clip_box', 'grid_clip_on', 'grid_clip_path', 'grid_color', 'grid_dash_capstyle', 'grid_dash_joinstyle', 'grid_dashes', 'grid_data', 'grid_drawstyle', 'grid_figure', 'grid_fillstyle', 'grid_gid', 'grid_in_layout', 'grid_label', 'grid_linestyle', 'grid_linewidth', 'grid_marker', 'grid_markeredgecolor', 'grid_markeredgewidth', 'grid_markerfacecolor', 'grid_markerfacecoloralt', 'grid_markersize', 'grid_markevery', 'grid_path_effects', 'grid_picker', 'grid_pickradius', 'grid_rasterized', 'grid_sketch_params', 'grid_snap', 'grid_solid_capstyle', 'grid_solid_joinstyle', 'grid_transform', 'grid_url', 'grid_visible', 'grid_xdata', 'grid_ydata', 'grid_zorder', 'grid_aa', 'grid_c', 'grid_ds', 'grid_ls', 'grid_lw', 'grid_mec', 'grid_mew', 'grid_mfc', 'grid_mfcalt', 'grid_ms']
        ax.tick_params(axis='y', rotation=0, labelsize=22)
    #     if error_type == 'max error':
    #         ax.set_ylabel(f'Max Error', fontsize=fontsize)
    #     elif error_type == 'l1 error':
    #         ax.set_ylabel(f'Average Error', fontsize=fontsize)
    #     ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # plt.title(f'Input data is {dataname} and \nquery class is categorical-marginals')
    plt.show()


###############
## Fake data.
###############

# df = read_result('../ICML/ml_results/ML_LR.csv', ml_model='LR')
# df = read_result('../ICML/ml_results/ML_RF.csv', ml_model='RF')
df = read_result('../ICML/ml_results/ML_XGBoost.csv', ml_model='XGB')


# df = pd.concat([df1, df2, df3], ignore_index=True)
df = df.loc[(df['epsilon'] == 0.07), :]
# df = df.loc[(df[target_label] == 'PUBCOV'), :]
# df = df.loc[(df[target_label] == 'PINCP'), :]
df = df.loc[(df[target_label] == 'ESR'), :]
df[clf_error_label] = df[clf_error_label].round(4)
print(df)
# show_result(df)
