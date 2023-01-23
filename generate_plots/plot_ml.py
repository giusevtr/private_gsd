import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")
sns.set(font_scale=1.5)

# def read_result(path, error_type='max error'):
def read_result(data_path, algo_name, query_name, error_type='max error'):
    # data_path = f'{data_dir}/mix/privga/result_mix_privga.csv'
    df = pd.read_csv(data_path)
    df = df.rename(columns={"dataset": "data", "error_max": "max error", 'error_mean': 'l1 error',
                            "test_seed": "seed"})

    replace_names = ['-train', '-cat-train']
    for rn in replace_names:
        df['data'] = df['data'].replace(to_replace=f"folktables_2018_mobility_CA{rn}",
                                        value="folktables_2018_mobility_CA")

    df = df[['data',  'T', 'epsilon', 'seed', 'max error', 'l1 error']]

    df = df.groupby(['data', 'T', 'epsilon'], as_index=False)[error_type].mean()
    df = df.groupby(['data', 'epsilon'], as_index=False)[error_type].min()
    df['data'] = algo_name
    # privga_df['generator'] = 'PrivGA'
    return df


def show_result(df):


    # sns.barplot(data=df, x='generator', y='ML accuracy', hue='ML method type', row='epsilon')
    sns.catplot(data=df, x="generator", y='ML accuracy', hue='ML method type', row='epsilon', kind="bar")

    plt.subplots_adjust(top=0.9)

    # for ax in g.axes.flat:
    #     epsilon_vals = [0.07, 0.23, 0.52, 0.74, 1.00]
    #     ax.set_xticks(epsilon_vals)
    #     ax.set_xticklabels( rotation=0, labels=epsilon_vals)
    #     ax.set_xlabel(f'epsilon', fontsize=fontsize)
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
cols = ['data', 'generator', 'stats', 'T', 'epsilon', 'seed', 'ML method type', 'ML seed', 'ML accuracy']
res = [
    ['acsreal', 'Original', '', 0, 0.07, 0, 'RF', 0, 0.99],
    ['acsreal', 'PrivGA', 'prefix', 50, 0.07, 0, 'RF', 0, 0.96],
    ['acsreal', 'GEM', 'ranges', 50, 0.07, 0, 'RF', 0, 0.91],
    ['acsreal', 'RAP', 'ranges', 50, 0.07, 0, 'RF', 0, 0.95],

    ['acsreal', 'Original', '', 50, 0.07, 0, 'LG', 0, 0.93],
    ['acsreal', 'PrivGA', 'prefix', 50, 0.07, 0, 'LG', 0, 0.90],
    ['acsreal', 'GEM', 'ranges', 50, 0.07, 0, 'LG', 0, 0.81],
    ['acsreal', 'RAP', 'ranges', 50, 0.07, 0, 'LG', 0, 0.80],


    ['acsreal', 'Original', '', 0, 1.00, 0, 'RF', 0, 0.99],
    ['acsreal', 'PrivGA', 'prefix', 50, 1.00, 0, 'RF', 0, 0.98],
    ['acsreal', 'GEM', 'ranges', 50, 1.00, 0, 'RF', 0, 0.93],
    ['acsreal', 'RAP', 'ranges', 50, 1.00, 0, 'RF', 0, 0.97],

    ['acsreal', 'Original', '', 50, 1.00, 0, 'LG', 0, 0.93],
    ['acsreal', 'PrivGA', 'prefix', 50, 1.00, 0, 'LG', 0, 0.91],
    ['acsreal', 'GEM', 'ranges', 50, 1.00, 0, 'LG', 0, 0.91],
    ['acsreal', 'RAP', 'ranges', 50, 1.00, 0, 'LG', 0, 0.90],
]
df = pd.DataFrame(res, columns=cols)
show_result(df)