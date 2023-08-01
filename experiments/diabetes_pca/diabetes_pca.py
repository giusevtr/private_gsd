import pandas as pd
import os
from models import GSD
from utils import Dataset, Domain
import numpy as np
import seaborn as sns
import jax.random
from models import GSDtemp
from stats import ChainedStatistics,  Marginals, NullCounts
import jax.numpy as jnp
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

QUANTILES = 50
COMPONENTS = 7

ordinal = ['Pregnant',  # 17
           'Plasma',  # 199
           'BloodPressure',  # 122
           'Triceps', 'Age', 'Label']
columns = ['Pregnant',
           'Plasma',
           'BloodPressure',  # 122
           'Triceps',  # 99
           'Insulin',  # 846
           'BMI',  # 67.1
           'Diabetes_pedigree',  # 2.342
           'Age',  # 60
           'Label']


def get_pca_query(df):
    X_orig = df.values[:, :-1]
    avg = X_orig.mean(axis=0)
    std = X_orig.std(axis=0)

    X_train_normed = (X_orig - avg) / std

    pca = PCA(n_components=COMPONENTS)
    pca.fit(X_train_normed)
    c = jnp.array(pca.components_)

    X_proj = np.dot(X_train_normed, c.T)

    quantiles = np.linspace(0, 1, QUANTILES)
    thresholds = jnp.array(np.quantile(X_proj, q=quantiles, axis=0))
    # print(c)
    print(pca.explained_variance_ratio_, np.sum(pca.explained_variance_ratio_))
    # print(pca.singular_values_)
    # print()

    def query(data):
        n = data.shape[0]
        # separate X,y
        X = data[:, :-1]
        y = data[:, -1].reshape((-1, 1))

        # normalize features
        X_norm = (X - avg) / std
        proj = jnp.dot(X_norm, c.T)

        below_threshold = (proj < thresholds.reshape(QUANTILES, 1, COMPONENTS))
        # a = below_threshold.reshape((n, -1))
        a0 = below_threshold * y
        a1 = below_threshold * (1-y)

        stats0 = a0.mean(axis=1).flatten()
        stats1 = a1.mean(axis=1).flatten()
        stats = jnp.concatenate((stats0, stats1))
        return stats

    return query


def get_preprocess_df(df, ordinal):
    columns = df.columns
    preprocess = {}
    config = {}
    X_pre = []
    for col in columns:
        if col in ordinal:
            minv, maxv = df[col].values.min(), df[col].values.max()
            ran = maxv - minv
            preprocess[col] = (minv, maxv)
            config[col] = {"type": "ordinal", "size": int(ran) + 1}
            col_pre = (df[col].values - minv).astype(int)
            X_pre.append(pd.Series(col_pre, name=col))
        else:
            minv, maxv = df[col].values.min(), df[col].values.max()
            ran = maxv - minv
            preprocess[col] = (minv, maxv)
            config[col] = {"type": "numerical", "size": 1}
            col_pre = (df[col].values - minv) / ran
            X_pre.append(pd.Series(col_pre, name=col))

    df = pd.concat(X_pre, axis=1)
    return df, preprocess, config

def plot_covar(data_df: pd.DataFrame):
    temp = data_df[['Plasma', 'BloodPressure', 'Triceps', 'Insulin', 'BMI', 'Age']]
    print(temp.cov())

def plot_hist(real_df, sync_df):
    # cols = ['Plasma', 'BloodPressure', 'Triceps', 'Insulin', 'BMI', 'Age']
    lbl = 'Label'
    for col in columns[:-1]:
        real_temp = real_df[[col, lbl]]
        sync_temp = sync_df[[col, lbl]]
        real_temp.loc[:, ['Type']] = 'Real'
        sync_temp.loc[:, ['Type']] = 'Sync'
        temp = pd.concat((real_temp, sync_temp), ignore_index=True)
        # sns.histplot(temp, x=col, hue='Type')
        g = sns.FacetGrid(temp, col=lbl, hue='Type')
        g.map(sns.histplot, col)

        plt.show()

if __name__ == "__main__":
    for seed in [0]:
        data_dir = '../data2/data'
        X_train = np.load(f'{data_dir}/diabetes/kfolds/{seed}/X_num_train.npy')
        X_test = np.load(f'{data_dir}/diabetes/kfolds/{seed}/X_num_test.npy')
        X_val = np.load(f'{data_dir}/diabetes/kfolds/{seed}/X_num_val.npy')
        y_train = np.load(f'{data_dir}/diabetes/kfolds/{seed}/y_train.npy')
        y_test = np.load(f'{data_dir}/diabetes/kfolds/{seed}/y_test.npy')
        y_val = np.load(f'{data_dir}/diabetes/kfolds/{seed}/y_val.npy')

        X = np.concatenate((X_train, X_test, X_val))
        y = np.concatenate((y_train, y_test, y_val))


        df = pd.DataFrame(np.column_stack((X_train, y_train)), columns=columns)
        df_pre, preprocess, config = get_preprocess_df(df, ordinal)

        bins_edges = {}

        quantiles = np.linspace(0, 1, QUANTILES)
        lo = []
        hi = []
        avg = []
        std = []
        for att in config:
            v = df_pre[att].values
            if config[att]['type'] == 'numerical':
                thresholds = np.quantile(v, q=quantiles)
                bins_edges[att] = thresholds
            lo.append(v.min())
            hi.append(v.max())
            avg.append(v.mean())
            std.append(v.std())
        avg = jnp.array(avg)
        std = jnp.array(std)

        ranges = jnp.array(hi)

        # df_pre = pd.DataFrame(np.column_stack((X_pre, y_train)), columns=columns)
        domain = Domain(config, bin_edges=bins_edges)
        # domain = Domain(config)
        data = Dataset(df_pre, domain)
        print()



        modules = []
        modules.append(Marginals.get_all_kway_combinations(data.domain, k=2, levels=1, bin_edges=bins_edges,
                                                           include_feature='Label'))
        stat_module = ChainedStatistics(modules)
        stat_module.fit(data, max_queries_per_workload=30000)

        stat_fn0 = stat_module._get_workload_fn()

        key = jax.random.PRNGKey(seed)
        N = len(data.df)
        algo = GSDtemp(num_generations=600000,
                           print_progress=True,
                           stop_early=True,
                           domain=data.domain,
                           population_size=50,
                           data_size=N,
                           stop_early_gen=3* N)

        def means_stat_fn(X):
            # X2 = (X - avg) / std
            # covar = (jnp.dot(X2.T, X) / X2.shape[0]).flatten()
            means = X.mean(axis=0)
            # return jnp.concatenate((means, covar))
            return means

        pca_stat_fn = get_pca_query(df_pre)


        def stat_fn(X):
            return jnp.concatenate((
                    # pca_stat_fn(X),
                    stat_fn0(X),
                ))
            # return stat1


        true_stats = stat_fn(data.to_numpy())

        # delta = 1.0 / len(data) ** 2
        sync_data_pre = algo.fit_function(key, true_stats, stat_fn)


        sync_data = sync_data_pre.df.copy()
        post = []
        # POST
        for col in columns:
            minv, maxv = preprocess[col]
            ran = maxv - minv
            if col in ordinal:
                col_df = sync_data_pre.df[col] + minv
                post.append(col_df)
            else:
                col_df = (sync_data_pre.df[col] * ran + minv)
                post.append(col_df)


        post_sync_df = pd.concat(post, axis=1)

        dataset_name = 'diabetes'
        k=3
        eps=100000
        data_size_str = 'N'
        sync_dir = f'sync_data/{dataset_name}/{k}/{eps:.2f}/{data_size_str}/oneshot'
        os.makedirs(sync_dir, exist_ok=True)
        print(f'Saving {sync_dir}/sync_data_{seed}.csv')
        post_sync_df.to_csv(f'{sync_dir}/sync_data_{seed}.csv', index=False)

        errors = jnp.abs(true_stats - stat_fn(sync_data.to_numpy()))

        print(f'Input data {dataset_name}, k={k}, epsilon={eps:.2f}, seed={seed}')
        print(f'GSD(oneshot):'
              f'\t max error = {errors.max():.5f}'
              f'\t avg error = {errors.mean():.6f}')
        plot_hist(df, post_sync_df)

        plot_covar(df)
        plot_covar(post_sync_df)
        print('Mean:')
        for col in columns:
            print(f'{df[col].mean():<10.4f}\t{post_sync_df[col].mean():<10.4f}')
        print('Std:')
        for col in columns:
            print(f'{df[col].std():<10.4f}\t{post_sync_df[col].std():<10.4f}')
