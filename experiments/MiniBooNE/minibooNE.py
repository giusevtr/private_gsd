import pandas as pd
import os
from utils import Dataset, Domain
import jax.random
from models import GSD
import jax.numpy as jnp
from stats.thresholds import get_thresholds_realvalued
from stats import ChainedStatistics,  Marginals, NullCounts
from dp_data import cleanup, DataPreprocessor, get_config_from_json
from experiments.utils import read_original_data, read_tabddpm_data
from utils import timer


dataset_name = 'miniboone'
k=2
train_df, val_df, test_df, all_cols, cat_cols, num_cols = read_original_data(dataset_name)
all_df = pd.concat([train_df, val_df, test_df])
config = get_config_from_json({'categorical': cat_cols + ['Label'], 'ordinal': [], 'numerical': num_cols})
preprocessor = DataPreprocessor(config=config)
preprocessor.fit(all_df)
pre_train_df = preprocessor.transform(train_df)
pre_val_df = preprocessor.transform(val_df)
pre_test_df = preprocessor.transform(test_df)

N = len(train_df)
min_bin_size = N // 200  # Number of points on each edge
bin_edges = {}
confi_dict = preprocessor.get_domain()
for col_name in num_cols:
    bin_edges[col_name] = get_thresholds_realvalued(pre_train_df[col_name], min_bin_size, levels=20)
    print(f'{col_name}: Edges = {len(bin_edges[col_name])}')

domain = Domain(confi_dict, bin_edges=bin_edges)
data = Dataset(pre_train_df, domain)
X = data.to_numpy()
N = len(data.df)

df0 = pre_train_df[pre_train_df['Label'] == 0]
df1 = pre_train_df[pre_train_df['Label'] == 1]
orig_cnt_0 = jnp.array([len(df0)])
orig_cnt_1 = jnp.array([len(df1)])
orig_means_0 = (jnp.array(df0.values[:, 1:].sum(axis=0)) / orig_cnt_0).reshape((-1))
orig_means_1 = (jnp.array(df1.values[:, 1:].sum(axis=0)) / orig_cnt_1).reshape((-1))
orig_var_0 = jnp.array(df0.values[:, 1:].var(axis=0))
orig_var_1 = jnp.array(df1.values[:, 1:].var(axis=0))

for data_size_str in ['2000']:
    for seed in [0]:
        modules = []
        modules.append(Marginals.get_all_kway_combinations(data.domain, k=k, levels=1, bin_edges=bin_edges,
                                                        include_feature='Label'))
        stat_module = ChainedStatistics(modules)
        stat_module.fit(data, max_queries_per_workload=10000)

        # true_stats = stat_module.get_all_true_statistics()
        stat_fn0 = stat_module._get_workload_fn()

        def stat_fn_covar(X):
            # stats0 = stat_fn0(X)
            n = X.shape[0]
            y = X[:, 0].reshape((-1, 1))
            cnt_0 = (y==0).sum()
            cnt_1 = (y==1).sum()

            # idx0 = jnp.argwhere(y == 0).flatten()
            # idx1 = jnp.argwhere(y == 1).flatten()
            # X0 = X[idx0, 1:]
            # X1 = X[idx1, 1:]
            X0 = jnp.multiply(X[:, 1:], y == 0)
            X1 = jnp.multiply(X[:, 1:], y == 1)
            F0 = jnp.multiply((X[:, 1:] - orig_means_0) , y == 0)
            F1 = jnp.multiply((X[:, 1:] - orig_means_1), y == 1)

            means_0 = X0.sum(axis=0)
            means_1 = X1.sum(axis=0)
            std_0 = ((X0 - orig_means_0) ** 2) .sum(axis=0)
            std_1 = ((X1 - orig_means_1) ** 2) .sum(axis=0)

            c0 = jnp.dot(F0.T, F0).flatten()
            c1 = jnp.dot(F1.T, F1).flatten()
            ans = jnp.concatenate((cnt_0.reshape((1,)), cnt_1.reshape((1,)),
                                   means_0, means_1,
                                   # std_0, std_1,
                                   c0, c1
                                   )
                                  )
            return ans / n
        true_stats = stat_fn_covar(data.to_numpy())
        print(f'true_stats.shape = ', true_stats.shape)

        key = jax.random.PRNGKey(seed)
        # N = len(data.df)
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
        sync_data = algo.fit_help(key, true_stats, stat_fn_covar)
        # sync_data = algo.fit_zcdp(key, stat_module=stat_module, rho=1000000000)

        sync_data_df = sync_data.df.copy()
        sync_data_df_post = preprocessor.inverse_transform(sync_data_df)

        sync_dir = f'sync_data/{dataset_name}/{k}_covar/inf/{data_size_str}/oneshot'
        os.makedirs(sync_dir, exist_ok=True)
        print(f'Saving {sync_dir}/sync_data_{seed}.csv')
        sync_data_df_post.to_csv(f'{sync_dir}/sync_data_{seed}.csv', index=False)

        errors = jnp.abs(true_stats - stat_fn_covar(sync_data.to_numpy()))
        print(f'Input data {dataset_name}, k={k}, epsilon=inf, seed={seed}')
        print(f'GSD(oneshot):'
              f'\t max error = {errors.max():.5f}'
              f'\t avg error = {errors.mean():.6f}')
        etime = timer(stime, msg='Total time:')
        f = open(f'{sync_dir}/time.txt', 'w')
        elapsed = etime - stime
        f.write(str(elapsed))
        f.close()

