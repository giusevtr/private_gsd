import os.path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dp_data import load_domain_config, load_df, get_evaluate_ml
from utils import timer, Dataset, Domain


def filter_outliers(df_train, df_test):
    domain = Domain.fromdict(config)

    df = df_train.append(df_test)

    for num_col in domain.get_numeric_cols():
        q_lo = df[num_col].quantile(0.01)
        q_hi = df[num_col].quantile(0.99)
        size1 = len(df_train)
        df_train_filtered = df_train[(df_train[num_col] <= q_hi) & (df_train[num_col] >= q_lo)]
        df_test_filtered = df_test[(df_test[num_col] <= q_hi) & (df_test[num_col] >= q_lo)]
        size2 = len(df_train_filtered)
        print(f'Numeric column={num_col}. Removing {size1 - size2} rows.')
        # df_filtered = df[(df[num_col] <= q_hi)]
        df_train = df_train_filtered
        df_test = df_test_filtered

    for num_col in domain.get_numeric_cols():
        maxv = df_train[num_col].max()
        minv = df_train[num_col].min()
        meanv = df_train[num_col].mean()
        print(f'Col={num_col:<10}: mean={meanv:<5.3f}, min={minv:<5.3f},  max={maxv:<5.3f},')
        df_train[num_col].hist()
        plt.yscale('log')
        plt.show()

    return df_train, df_test



def print_mean(df_real, df_sync, config):
    domain = Domain.fromdict(config)

    num_cols = domain.get_numeric_cols()
    real_target = df_real['PINCP']
    sync_target = df_sync['PINCP']

    print('real:')
    print(df_real[num_cols + ['PINCP']].corr())
    print('sync:')
    print(df_sync[num_cols + ['PINCP']].corr())
    print(f'{"col":<7}: {"real mean":<10}|{"sync mean":<10}||{"real std":<10}|{"sync std":<10}')
    # return 0
    for c in domain.get_numeric_cols():
        mean_real = df_real[c].mean()
        std_real = df_real[c].std()
        mean_sync = df_sync[c].mean()
        std_sync = df_sync[c].std()
        print(f'{c:<7}: {mean_real:<10.4f}|{mean_sync:<10.4f}||{std_real:<10.4f}|{std_sync:<10.4f}')


        real = df_real[[c, 'PINCP']]
        real.loc[:, 'Type'] = 'Real'
        sync = df_sync[[c, 'PINCP']]
        sync.loc[:, 'Type'] = 'Sync'

        df = pd.concat([real.sample(n=2000), sync])
        # sns.histplot(data=df, x=c, hue='PINCP')
        # sns.histplot(data=df, x=c, hue='Type')

        g = sns.FacetGrid(df, row='PINCP', hue='Type')
        g.map(plt.hist, c, alpha=0.5)
        g.set(yscale='log')

        plt.show()

        print()





if __name__ == "__main__":
    algo = 'RAP'
    module = 'Ranges'
    # epochs = [3, 4, 5, 6, 7, 8, 9]


    # algo = 'PrivateGSD'
    epochs = [3, 4, 5, 6, 7, 8, 9, 10, 20, 25, 40, 50, 60, 75, 80, 100]
    # module = 'Prefix'
    # module = 'Halfspaces'

    epsilon_vals = [0.07, 0.23, 0.52, 0.74, 1]
    seeds = [0, 1, 2]


    dataset_name = 'folktables_2018_real_NY'
    root_path = '../../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)
    df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
    df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')
    targets = ['PINCP',  'PUBCOV', 'ESR']

    ml_eval_fn = get_evaluate_ml(df_test, config, targets=targets, models=['LogisticRegression'])

    orig_results = ml_eval_fn(df_train.sample(n=20000), 0 )
    print(orig_results)
    sync_dir = f'../../sync_data/{algo}/{module}'

    Res = []
    for e in epochs:
        for eps in epsilon_vals:
            for seed in seeds:
                sync_path = f'{sync_dir}/{e}/{eps:.2f}/sync_data_{seed}.csv'
                if not os.path.exists((sync_path)): continue
                df_sync = pd.read_csv(sync_path, index_col=None)
                results = ml_eval_fn(df_sync, seed)
                # results = results[results['Metric'] == 'f1_macro']
                results = results[results['Eval Data'] == 'Test']
                results['Epoch'] = e
                results['Epsilon'] = eps
                results['Seed'] = seed
                Res.append(results)
                print()
                print(results)
                if eps == 1.0 and e == 80:
                    print_mean(df_train, df_sync, config)

    all_results = pd.concat(Res)

    all_results.to_csv(f'acsreal_results_{algo}_{module}.csv')

    # df_sync1 = pd.read_csv('folktables_2018_multitask_NY_sync_0.07_0.csv')
    # df_sync2 = pd.read_csv('folktables_2018_multitask_NY_sync_1.00_0.csv')
    #
    # print('Synthetic train eps=0.07:')
    # results = ml_eval_fn(df_sync1, 0)
    # results = results[results['Metric'] == 'f1']
    # print(results)
    #
    # print('Synthetic train eps=1.00:')
    # results = ml_eval_fn(df_sync2, 0)
    # results = results[results['Metric'] == 'f1']
    # print(results)
