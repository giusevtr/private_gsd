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
import seaborn as sns
from stats.thresholds import get_thresholds_ordinal
from stats.get_marginals_fn import get_marginal_query

from stats import ChainedStatistics,  Marginals, NullCounts
from dp_data import cleanup, DataPreprocessor, get_config_from_json
QUANTILES = 50
from experiments.run_exp import run_experiment


def get_constraint_fn(domain):
    # Build  constraint
    race_feat = 'cat_5'
    sex_feat = 'cat_6'
    sex_att_ind = domain.get_attribute_indices([sex_feat])[0]
    label_ind = domain.get_attribute_indices(['Label'])[0]
    def constraint_fn(X):
        males = (X[:, sex_att_ind] == 1).mean()
        loss1 = jnp.abs(males - 0.5)**2

        males_inc = ((X[:, sex_att_ind] == 1) & (X[:, label_ind] == 1)).mean()
        females_inc = ((X[:, sex_att_ind] == 0) & (X[:, label_ind] == 1)).mean()
        loss2 = (males_inc - females_inc)**2
        return loss1 + loss2

run_experiment(data_name='adult', data_size_str='1000', k=2, )

# data_name = 'adult'
# for seed in range(1):
#     k = 2
#     data_size_str = '1000'
#
#     data_path = '../data2/data'
#     X_num_train = np.load(f'{data_path}/{data_name}/X_num_train.npy').astype(int)
#     X_num_val = np.load(  f'{data_path}/{data_name}/X_num_val.npy').astype(int)
#     X_num_test = np.load( f'{data_path}/{data_name}/X_num_test.npy').astype(int)
#     X_cat_train = np.load(f'{data_path}/{data_name}/X_cat_train.npy')
#     X_cat_val = np.load(  f'{data_path}/{data_name}/X_cat_val.npy')
#     X_cat_test = np.load( f'{data_path}/{data_name}/X_cat_test.npy')
#     y_train = np.load(    f'{data_path}/{data_name}/y_train.npy')
#     y_val = np.load(      f'{data_path}/{data_name}/y_val.npy')
#     y_test = np.load(     f'{data_path}/{data_name}/y_test.npy')
#
#     cat_cols = [f'cat_{i}' for i in range(X_cat_train.shape[1])]
#     num_cols = [f'num_{i}' for i in range(X_num_train.shape[1])]
#     all_cols = cat_cols + num_cols + ['Label']
#
#     train_df = pd.DataFrame(np.column_stack((X_cat_train, X_num_train, y_train)), columns=all_cols)
#     val_df = pd.DataFrame(np.column_stack((X_cat_val, X_num_val, y_val)), columns=all_cols)
#     test_df = pd.DataFrame(np.column_stack((X_cat_test, X_num_test, y_test)), columns=all_cols)
#
#
#
#     all_df = pd.concat([train_df, val_df, test_df])
#
#     config = get_config_from_json({'categorical': cat_cols + ['Label'], 'ordinal': num_cols, 'numerical': []})
#     preprocessor = DataPreprocessor(config=config)
#     preprocessor.fit(all_df)
#     pre_train_df = preprocessor.transform(train_df)
#     pre_val_df = preprocessor.transform(val_df)
#     pre_test_df = preprocessor.transform(test_df)
#
#
#     N = len(train_df)
#     N_prime = N if data_size_str == 'N' else int(data_size_str)
#
#     min_bin_size = N // 200 # Number of points on each edge
#     # bin_edges = {}
#     confi_dict = preprocessor.get_domain()
#
#     # Create dataset and k-marginal queries
#     domain = Domain(confi_dict)
#
#     data = Dataset(pre_train_df, domain)
#
#     true_stats, stat_fn, total_error_fn = get_marginal_query(data, domain, k=k,
#                                              min_bin_density=0.005,
#                                             minimum_density=0.999,
#                                             max_marginal_size=N_prime,
#                                              include_features=['Label'], verbose=True)
#
#     true_stats2 = stat_fn(data.to_numpy())
#     print(f'Max error: ', jnp.abs(true_stats2 - true_stats).max())
#     print(f'Avg error: ', jnp.abs(true_stats2 - true_stats).mean())
#
#
#
#     key = jax.random.PRNGKey(seed)
#
#     algo = GSD(num_generations=30000,
#                print_progress=True,
#                stop_early=True,
#                domain=data.domain,
#                population_size=50,
#                data_size=N_prime,
#                stop_early_gen=N_prime,
#                sparse_statistics=True
#                )
#     # delta = 1.0 / len(data) ** 2
#     sync_data = algo.fit_help(key, true_stats, stat_fn,
#                               # constraint_fn=constraint_fn
#                               )
#
#     sync_data_df = sync_data.df.copy()
#     sync_data_df_post = preprocessor.inverse_transform(sync_data_df)
#
#
#     sync_dir = f'sync_data/{data_name}/{k}/inf/{data_size_str}/oneshot'
#     os.makedirs(sync_dir, exist_ok=True)
#     print(f'Saving {sync_dir}/sync_data_{seed}.csv')
#     sync_data_df_post.to_csv(f'{sync_dir}/sync_data_{seed}.csv', index=False)
#
#     errors = jnp.abs(true_stats - stat_fn(sync_data.to_numpy()))
#
#     print(f'Input data {data_name}, k={k}, seed={seed}')
#     print(f'GSD(oneshot):'
#           f'\t max error = {errors.max():.5f}'
#           f'\t avg error = {errors.mean():.6f}')
#     print(f'constraint: ', constraint_fn(sync_data.to_numpy()))
#     print(f'Train:')
#     print(train_df[sex_feat].value_counts())
#     print('Sync:')
#     print(sync_data_df_post[sex_feat].value_counts())
#
#     g = sns.FacetGrid(train_df[[sex_feat, 'Label']], col=sex_feat, col_order=['Male', 'Female'])
#     g.map(sns.histplot, 'Label')
#     plt.show()
#
#     g = sns.FacetGrid(sync_data_df_post[[sex_feat, 'Label']], col=sex_feat, col_order=['Male', 'Female'])
#     g.map(sns.histplot, 'Label')
#     plt.show()
#     print()
#     total_errors_df = total_error_fn(data, sync_data)
#     print(total_errors_df['Max'].max())
#     print(total_errors_df['Average'].mean())
#     print(total_errors_df)
#

