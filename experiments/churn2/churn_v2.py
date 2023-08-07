import pandas as pd
import os
from utils import Dataset, Domain
import jax.random
from models import GeneticSDConsistent as GeneticSD
from models import GSD
import jax.numpy as jnp
from stats.thresholds import get_thresholds_ordinal, get_thresholds_realvalued
from experiments.utils_for_experiment import read_original_data, read_tabddpm_data
from stats import ChainedStatistics,  Marginals, NullCounts
from dp_data import cleanup, DataPreprocessor, get_config_from_json


dataset_name = 'churn2'
k = 3
eps = 100000
data_size_str = 'N'

# real_cols = [f'num_3', 'num_5', 'num_6']
# ord_cols = ['num_0', 'num_1', 'num_2', 'num_4']
train_df, test_df, all_df, cat_cols, ord_cols, real_cols = read_original_data(dataset_name, root_dir='../data2/data')


config = get_config_from_json({'categorical': cat_cols + ['Label'], 'ordinal': ord_cols, 'numerical': real_cols})
preprocessor = DataPreprocessor(config=config)
preprocessor.fit(all_df)
pre_train_df = preprocessor.transform(train_df)
pre_test_df = preprocessor.transform(test_df)


N = len(train_df)
min_bin_size = N // 200 # Number of points on each edge
bin_edges = {}
confi_dict = preprocessor.get_domain()
for col_name in ord_cols:
    sz = confi_dict[col_name]['size']
    bin_edges[col_name] = get_thresholds_ordinal(pre_train_df[col_name], min_bin_size, sz,
                                                 levels=20)
for col_name in real_cols:
    bin_edges[col_name] = get_thresholds_realvalued(pre_train_df[col_name], min_bin_size,
                                                 levels=20)
    print()
domain = Domain(confi_dict, bin_edges=bin_edges)
data = Dataset(pre_train_df, domain)

for seed in [0, 1, 2, 3, 4]:

    modules = []
    modules.append(Marginals.get_all_kway_combinations(data.domain, k=3, levels=1, bin_edges=bin_edges,
                                                       include_feature='Label'))
    stat_module = ChainedStatistics(modules)
    stat_module.fit(data, max_queries_per_workload=2000)

    true_stats = stat_module.get_all_true_statistics()
    stat_fn0 = stat_module._get_workload_fn()

    key = jax.random.PRNGKey(seed)
    N = len(data.df)
    algo = GSD(num_generations=1000000,
               print_progress=False,
               stop_early=True,
               domain=data.domain,
               population_size=50,
               data_size=N,
               stop_early_gen=N,
               sparse_statistics=True
               )
    sync_data = algo.fit_zcdp(key, stat_module=stat_module, rho=1000000000)
    sync_data_df = sync_data.df.copy()
    sync_data_df_post = preprocessor.inverse_transform(sync_data_df)

    sync_dir = f'sync_data/{dataset_name}/{k}/{eps:.2f}/{data_size_str}/oneshot'
    os.makedirs(sync_dir, exist_ok=True)
    print(f'Saving {sync_dir}/sync_data_{seed}.csv')
    sync_data_df_post.to_csv(f'{sync_dir}/sync_data_{seed}.csv', index=False)
    errors = jnp.abs(true_stats - stat_fn0(sync_data.to_numpy()))
    print(f'Input data {dataset_name}, k={k}, epsilon={eps:.2f}, seed={seed}')
    print(f'GSD(oneshot):'
          f'\t max error = {errors.max():.5f}'
          f'\t avg error = {errors.mean():.6f}')

