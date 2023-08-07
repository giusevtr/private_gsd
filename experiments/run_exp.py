
import pandas as pd
import os
from models import GSD
from utils import Dataset, Domain, timer
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
from experiments.utils_for_experiment import read_original_data, read_tabddpm_data


def run_experiment(data_name, data_size_str = 'N', k=2, protected_feature=None, seeds=(0,)):
    train_df, test_df, all_df, cat_cols, ord_cols, real_cols = read_original_data(data_name, root_dir='../data2/data')
    print(f'Categorical columns =',cat_cols)
    print(f'Ordinal columns =', ord_cols)
    print(f'Real-valued columns =', real_cols)


    config = get_config_from_json({'categorical': cat_cols + ['Label'], 'ordinal': ord_cols, 'numerical': real_cols})
    preprocessor = DataPreprocessor(config=config)
    preprocessor.fit(all_df)
    pre_train_df = preprocessor.transform(train_df)

    N = len(train_df)
    N_prime = N if data_size_str == 'N' else int(data_size_str)

    # Create dataset and k-marginal queries
    config_dict = preprocessor.get_domain()
    domain = Domain(config_dict)
    data = Dataset(pre_train_df, domain)

    algo = GSD(num_generations=1000000,
                   print_progress=False,
                   stop_early=True,
                   domain=data.domain,
                   population_size=50,
                   data_size=N_prime,
                   stop_early_gen=N_prime,
                   )
        # delta = 1.0 / len(data) ** 2
    for seed in list(seeds):
        key = jax.random.PRNGKey(seed)

        t0 = timer()
        true_stats, stat_fn, total_error_fn = get_marginal_query(seed, data, domain, k=k,
                                                                 min_bin_density=0.005,
                                                                 minimum_density=0.999,
                                                                 max_marginal_size=N_prime,
                                                                 include_features=['Label'], verbose=True)

        sync_data = algo.fit_help(key, true_stats, stat_fn)
        sync_data_df = sync_data.df.copy()
        sync_data_df_post = preprocessor.inverse_transform(sync_data_df)

        stats_name = f'{k}' + protected_feature
        sync_dir = f'sync_data/{data_name}/{stats_name}/{data_size_str}/inf/{seed}'
        os.makedirs(sync_dir, exist_ok=True)
        print(f'Saving {sync_dir}/sync_data.csv')
        sync_data_df_post.to_csv(f'{sync_dir}/sync_data.csv', index=False)
        errors = jnp.abs(true_stats - stat_fn(sync_data.to_numpy()))

        elapsed_time = int(timer() - t0)
        f = open(f"{sync_dir}/time.txt", "w")
        f.write(f'{elapsed_time}')
        f.close()

        print(f'Input data {data_name}, k={k}, seed={seed}')
        print(f'GSD(oneshot):'
              f'\t max error = {errors.max():.5f}'
              f'\t avg error = {errors.mean():.6f}')

        total_errors_df = total_error_fn(data, sync_data)
        print('\tTotal max error = ', total_errors_df['Max'].max())
        print('\tTotal avg error = ', total_errors_df['Average'].mean())
        print(total_errors_df)
        total_errors_df.to_csv(f'{sync_dir}/errors.csv')
