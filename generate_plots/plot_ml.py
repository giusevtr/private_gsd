import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")
sns.set(font_scale=1.5)

# def read_result(path, error_type='max error'):
def read_result(data_path, ml_model):
    # data_path = f'{data_dir}/mix/privga/result_mix_privga.csv'
    df = pd.read_csv(data_path, index_col=0)
    df = df.rename(columns={"algo":"generator"})
    df['model'] = ml_model
    df['target'] = df['error type'].apply(lambda s: s.split()[0])

    df = df[[ 'generator', 'T', 'epsilon', 'seed', 'target', 'model', 'original accuracy', 'private accuracy']]
    df['classification error'] = 1 - df['private accuracy']

    # df.melt(id_vars=['data',  'algo', 'T', 'epsilon', 'seed'], )
    df = df.groupby(['generator', 'T', 'epsilon', 'target', 'model', 'original accuracy'], as_index=False)['classification error'].mean()
    df = df.groupby(['generator', 'epsilon', 'target', 'model', 'original accuracy'], as_index=False)['classification error'].max()
    # privga_df['generator'] = 'PrivGA'
    return df


def show_result(df):


    # sns.barplot(data=df, x='generator', y='ML accuracy', hue='ML method type', row='epsilon')
    g = sns.catplot(data=df,
                    x="model",
                    y='classification error',
                    hue='generator', row='epsilon', kind="bar",
                    col='target'
                )

    g.set_xticklabels( rotation=-60)
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

df1 = read_result('../ICML/ml_results/ML_RF.csv', ml_model='RF')
df2 = read_result('../ICML/ml_results/ML_LR.csv', ml_model='LR')
df3 = read_result('../ICML/ml_results/ML_XGBoost.csv', ml_model='XGB')

df = pd.concat([df1, df2, df3])
df = df.loc[(df['epsilon'] == 1) | (df['epsilon'] == 0.07), :]
show_result(df)