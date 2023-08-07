import pandas as pd
import os
from models import GSD
from utils import Dataset, Domain
import numpy as np

import jax.random
from models import GeneticSDConsistent as GeneticSD
from models import GSD
from stats import ChainedStatistics,  Marginals, NullCounts
import jax.numpy as jnp

from stats.get_marginals_fn import get_marginal_query
QUANTILES = 50

for seed in [0, 1, 2, 3, 4]:

    data_path = '../data2/data'
    X_train = np.load(f'{data_path}/diabetes/kfolds/{seed}/X_num_train.npy')
    X_test = np.load( f'{data_path}/diabetes/kfolds/{seed}/X_num_test.npy')
    X_val = np.load(  f'{data_path}/diabetes/kfolds/{seed}/X_num_val.npy')
    y_train = np.load(f'{data_path}/diabetes/kfolds/{seed}/y_train.npy')
    y_test = np.load( f'{data_path}/diabetes/kfolds/{seed}/y_test.npy')
    y_val = np.load(  f'{data_path}/diabetes/kfolds/{seed}/y_val.npy')

    X = np.concatenate((X_train, X_test, X_val))
    y = np.concatenate((y_train, y_test, y_val))

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
    df = pd.DataFrame(np.column_stack((X_train, y_train)), columns=columns)
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
    # config = get_config_from_json({'categorical': cat_cols + ['Label'], 'ordinal': num_cols, 'numerical': []})
    # preprocessor = DataPreprocessor(config=config)
    # preprocessor.fit(all_df)
    # pre_train_df = preprocessor.transform(train_df)
    # pre_val_df = preprocessor.transform(val_df)
    # pre_test_df = preprocessor.transform(test_df)
    df = pd.concat(X_pre, axis=1)

    bins_edges = {}

    quantiles = np.linspace(0, 1, QUANTILES)
    for att in config:
        if config[att]['type'] == 'numerical':
            v = df[att].values
            thresholds = np.quantile(v, q=quantiles)
            bins_edges[att] = thresholds

    # df_pre = pd.DataFrame(np.column_stack((X_pre, y_train)), columns=columns)
    domain = Domain(config, bin_edges=bins_edges)
    data = Dataset(df, domain)
    print()

    modules = []
    modules.append(Marginals.get_all_kway_combinations(data.domain, k=3, levels=1, bin_edges=bins_edges,
                                                       include_features=['Label']))
    stat_module = ChainedStatistics(modules)
    stat_module.fit(data, max_queries_per_workload=30000)

    true_stats = stat_module.get_all_true_statistics()
    stat_fn0 = stat_module._get_workload_fn()

    key = jax.random.PRNGKey(seed)
    N = len(data.df)
    algo = GSD(num_generations=600000,
                       print_progress=False,
                       stop_early=True,
                       domain=data.domain,
                       population_size=50,
                       data_size=N,
                       stop_early_gen=N,
                       )
    # delta = 1.0 / len(data) ** 2
    sync_data_pre = algo.fit_zcdp(key, stat_module=stat_module, rho=1000000000)

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

    errors = jnp.abs(true_stats - stat_fn0(sync_data.to_numpy()))

    print(f'Input data {dataset_name}, k={k}, epsilon={eps:.2f}, seed={seed}')
    print(f'GSD(oneshot):'
          f'\t max error = {errors.max():.5f}'
          f'\t avg error = {errors.mean():.6f}')
