import os, sys
import jax.random
import pandas as pd
import numpy as np
from models import GeneticSDConsistent as GeneticSD
from models import GSD
from stats import ChainedStatistics,  Marginals, NullCounts
import jax.numpy as jnp
from dp_data import load_domain_config, load_df
from dp_data.data_preprocessor import DataPreprocessor
from utils import timer, Dataset, Domain


if __name__ == "__main__":

    # epsilon = [0.07, 0.15, 0.23, 0.52, 0.74, 1]
    epsilon = [100000]
    data_size_str = 'N'
    k = 2
    SEEDS = [0, 1, 2]
    QUANTILES = 30

    private_state = 'CA'
    public_states = ['TX', 'NY', 'PA']

    DATA = [
        # 'folktables_2018_coverage',
        # 'folktables_2018_mobility',
        # 'folktables_2018_employment',
        # 'folktables_2018_income',
        'folktables_2018_travel',
    ]
    for eps in epsilon:
        for dataset_name in DATA:
            for public_state in public_states:
                priv_dataset_name = f'{dataset_name}_{private_state}'
                pub_dataset_name = f'{dataset_name}_{public_state}'

                root_path = 'dp-data-dev/datasets/preprocessed/folktables/1-Year/'
                config = load_domain_config(priv_dataset_name, root_path=root_path)
                df_train = load_df(priv_dataset_name, root_path=root_path, idxs_path='seed0/train')
                df_test = load_df(priv_dataset_name, root_path=root_path, idxs_path='seed0/test')

                config_public = load_domain_config(pub_dataset_name, root_path=root_path)
                df_train_public = load_df(pub_dataset_name, root_path=root_path, idxs_path='seed0/train')
                df_test_public = load_df(pub_dataset_name, root_path=root_path, idxs_path='seed0/test')

                bins_edges = {}

                quantiles = np.linspace(0, 1, QUANTILES)
                for att in config:
                    if config_public[att]['type'] == 'numerical':
                        v = df_train_public[att].values
                        thresholds = np.quantile(v, q=quantiles)
                        bins_edges[att] = thresholds

                domain = Domain(config=config, bin_edges=bins_edges)
                data = Dataset(df_train, domain)
                N = len(data.df)

                if data_size_str == 'N':
                    data_size = len(df_train_public)
                else:
                    data_size = int(data_size_str)
                    df_train_public = df_train_public.sample(data_size)

                # domain_public = Domain(config=config_public, bin_edges=bins_edges)
                data_pub = Dataset(df_train_public, domain)
                public_data_size = len(df_train_public)
                print(f'Input data {dataset_name}, epsilon={eps:.2f}, data_size={data_size}, k={k} ')
                print(f'Public data state is {public_state}, with size {public_data_size}.')

                # Create statistics and evaluate
                key = jax.random.PRNGKey(0)
                # One-shot queries
                modules = []

                modules.append(Marginals.get_all_kway_combinations(data.domain, k=2, levels=1, bin_edges=bins_edges))
                stat_module = ChainedStatistics(modules)
                stat_module.fit(data, max_queries_per_workload=30000)

                true_stats = stat_module.get_all_true_statistics()
                stat_fn0 = stat_module._get_workload_fn()

                N = len(data.df)
                algo = GSD(num_generations=500000,
                                   print_progress=True,
                                   stop_early=True,
                                   domain=data.domain,
                                   population_size=40,
                                   data_size=public_data_size,
                                   stop_early_gen=data_size,
                                   )
                # Choose algorithm parameters

                # delta = 1.0 / len(data) ** 2
                delta = N**(-2)
                # Generate differentially private synthetic data with ADAPTIVE mechanism
                for seed in SEEDS:
                    key = jax.random.PRNGKey(seed)
                    t0 = timer()

                    sync_data = algo.fit_dp(key, stat_module=stat_module,
                                       epsilon=eps,
                                       delta=delta,
                                            init_data=data_pub)

                    sync_dir = f'sync_data_public/{pub_dataset_name}/{priv_dataset_name}/{k}/{eps:.2f}/{data_size_str}/oneshot'
                    os.makedirs(sync_dir, exist_ok=True)
                    print(f'Saving {sync_dir}/sync_data_{seed}.csv')
                    sync_data.df.to_csv(f'{sync_dir}/sync_data_{seed}.csv', index=False)
                    errors = jnp.abs(true_stats - stat_fn0(sync_data.to_numpy()))

                    print(f'Input data {dataset_name}, k={k}, epsilon={eps:.2f}, data_size={data_size}, seed={seed}')
                    print(f'GSD(oneshot):'
                          f'\t max error = {errors.max():.5f}'
                          f'\t avg error = {errors.mean():.6f}'
                          f'\t time = {timer() - t0:.4f}')

                print()