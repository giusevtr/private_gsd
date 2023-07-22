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
import matplotlib.pyplot as plt

from stats.thresholds import get_thresholds_ordinal

from stats import ChainedStatistics,  Marginals, NullCounts
from dp_data import cleanup, DataPreprocessor, get_config_from_json
QUANTILES = 50

data_name = 'adult'
for seed in [4]:

    X_num_train = np.load(f'../../dp-data-dev/data2/data/{data_name}/X_num_train.npy').astype(int)
    X_num_val = np.load(f'../../dp-data-dev/data2/data/{data_name}/X_num_val.npy').astype(int)
    X_num_test = np.load(f'../../dp-data-dev/data2/data/{data_name}/X_num_test.npy').astype(int)

    X_cat_train = np.load(f'../../dp-data-dev/data2/data/{data_name}/X_cat_train.npy')
    X_cat_val = np.load(f'../../dp-data-dev/data2/data/{data_name}/X_cat_val.npy')
    X_cat_test = np.load(f'../../dp-data-dev/data2/data/{data_name}/X_cat_test.npy')

    y_train = np.load(f'../../dp-data-dev/data2/data/{data_name}/y_train.npy')
    y_val = np.load(f'../../dp-data-dev/data2/data/{data_name}/y_val.npy')
    y_test = np.load(f'../../dp-data-dev/data2/data/{data_name}/y_test.npy')

    cat_cols = [f'cat_{i}' for i in range(X_cat_train.shape[1])]
    num_cols = [f'num_{i}' for i in range(X_num_train.shape[1])]
    all_cols = cat_cols + num_cols + ['Label']

    train_df = pd.DataFrame(np.column_stack((X_cat_train, X_num_train, y_train)), columns=all_cols)
    val_df = pd.DataFrame(np.column_stack((X_cat_val, X_num_val, y_val)), columns=all_cols)
    test_df = pd.DataFrame(np.column_stack((X_cat_test, X_num_test, y_test)), columns=all_cols)
    all_df = pd.concat([train_df, val_df, test_df])

    config = get_config_from_json({'categorical': cat_cols + ['Label'], 'ordinal': num_cols, 'numerical': []})
    preprocessor = DataPreprocessor(config=config)
    preprocessor.fit(all_df)
    pre_train_df = preprocessor.transform(train_df)
    pre_val_df = preprocessor.transform(val_df)
    pre_test_df = preprocessor.transform(test_df)


    N = len(train_df)
    min_bin_size = N // 200 # Number of points on each edge
    bin_edges = {}
    confi_dict = preprocessor.get_domain()
    for col_name in num_cols:
        values = pre_train_df[col_name].values.astype(int)
        sz = confi_dict[col_name]['size']

        bin_edges[col_name] = get_thresholds_ordinal(pre_train_df[col_name], min_bin_size, sz,
                                                     levels=20)
        print()

    domain = Domain(confi_dict, bin_edges=bin_edges)

    data = Dataset(pre_train_df, domain)

    modules = []
    modules.append(Marginals.get_all_kway_combinations(data.domain, k=3, levels=1, bin_edges=bin_edges,
                                                       include_feature='Label'))
    stat_module = ChainedStatistics(modules)
    stat_module.fit(data, max_queries_per_workload=2000)

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
               sparse_statistics=True
               )
    # delta = 1.0 / len(data) ** 2
    sync_data = algo.fit_zcdp(key, stat_module=stat_module, rho=1000000000)

    sync_data_df = sync_data.df.copy()
    sync_data_df_post = preprocessor.inverse_transform(sync_data_df)

    k = 3
    eps = 100000
    data_size_str = 'N'
    sync_dir = f'sync_data/{data_name}/{k}/{eps:.2f}/{data_size_str}/oneshot'
    os.makedirs(sync_dir, exist_ok=True)
    print(f'Saving {sync_dir}/sync_data_{seed}.csv')
    sync_data_df_post.to_csv(f'{sync_dir}/sync_data_{seed}.csv', index=False)

    errors = jnp.abs(true_stats - stat_fn0(sync_data.to_numpy()))

    print(f'Input data {data_name}, k={k}, epsilon={eps:.2f}, seed={seed}')
    print(f'GSD(oneshot):'
          f'\t max error = {errors.max():.5f}'
          f'\t avg error = {errors.mean():.6f}')

