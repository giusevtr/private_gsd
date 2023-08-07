import pandas as pd
import os
from utils import Dataset, Domain
import jax.random
from models import GSD
import jax.numpy as jnp
from stats.thresholds import get_thresholds_realvalued
from stats import ChainedStatistics,  Marginals, NullCounts
from dp_data import cleanup, DataPreprocessor, get_config_from_json
from experiments.utils_for_experiment import read_original_data, read_tabddpm_data
from utils import timer
from stats.get_marginals_fn import get_marginal_query


dataset_name = 'miniboone'
k=2
train_df, val_df, test_df, all_cols, cat_cols, num_cols = read_original_data(dataset_name, root_dir='../data2/data')
all_df = pd.concat([train_df, val_df, test_df])
config = get_config_from_json({'categorical': cat_cols + ['Label'], 'ordinal': [], 'numerical': num_cols})
preprocessor = DataPreprocessor(config=config)
preprocessor.fit(all_df)
pre_train_df = preprocessor.transform(train_df)
pre_val_df = preprocessor.transform(val_df)
pre_test_df = preprocessor.transform(test_df)

N = len(train_df)
min_bin_size = N // 200  # Number of points on each edge
# bin_edges = {}
confi_dict = preprocessor.get_domain()
# for col_name in num_cols:
#     bin_edges[col_name] = get_thresholds_realvalued(pre_train_df[col_name], min_bin_size, levels=20)
#     print(f'{col_name}: Edges = {len(bin_edges[col_name])}')

domain = Domain(confi_dict)
data = Dataset(pre_train_df, domain)

true_stats, stat_fn = get_marginal_query(data, domain, k=k, min_bin_density=0.005,
                                         min_marginal_size=600,
                                         include_features=['Label'], verbose=True)
X = data.to_numpy()
N = len(data.df)

# df0 = pre_train_df[pre_train_df['Label'] == 0]
# df1 = pre_train_df[pre_train_df['Label'] == 1]
# orig_cnt_0 = jnp.array([len(df0)])
# orig_cnt_1 = jnp.array([len(df1)])
# orig_means_0 = (jnp.array(df0.values[:, 1:].sum(axis=0)) / orig_cnt_0).reshape((-1))
# orig_means_1 = (jnp.array(df1.values[:, 1:].sum(axis=0)) / orig_cnt_1).reshape((-1))
# orig_var_0 = jnp.array(df0.values[:, 1:].var(axis=0))
# orig_var_1 = jnp.array(df1.values[:, 1:].var(axis=0))

for data_size_str in ['2000']:
    for seed in [0]:
        key = jax.random.PRNGKey(seed)
        N = len(data.df) if data_size_str == 'N' else int(data_size_str)
        algo = GSD(num_generations=200000,
                   print_progress=True,
                   stop_early=True,
                   domain=data.domain,
                   population_size=50,
                   data_size=N,
                   stop_early_gen=N,
                   sparse_statistics=True
                   )

        stime = timer()
        sync_data = algo.fit_help(key, true_stats, stat_fn)
        # sync_data = algo.fit_zcdp(key, stat_module=stat_module, rho=1000000000)

        sync_data_df = sync_data.df.copy()
        sync_data_df_post = preprocessor.inverse_transform(sync_data_df)

        sync_dir = f'sync_data/{dataset_name}/{k}_covar/inf/{data_size_str}/oneshot'
        os.makedirs(sync_dir, exist_ok=True)
        print(f'Saving {sync_dir}/sync_data_{seed}.csv')
        sync_data_df_post.to_csv(f'{sync_dir}/sync_data_{seed}.csv', index=False)

        errors = jnp.abs(true_stats - stat_fn(sync_data.to_numpy()))
        print(f'Input data {dataset_name}, k={k}, epsilon=inf, seed={seed}')
        print(f'GSD(oneshot):'
              f'\t max error = {errors.max():.5f}'
              f'\t avg error = {errors.mean():.6f}')
        etime = timer(stime, msg='Total time:')
        f = open(f'{sync_dir}/time.txt', 'w')
        elapsed = etime - stime
        f.write(str(elapsed))
        f.close()

