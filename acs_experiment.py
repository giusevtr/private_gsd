import itertools
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


inc_bins_pre = jnp.array([-10000, -100, -10, -5,
                         1,
                         5,
                         10, 50, 100, 200,
                         500, 700, 1000, 2000, 3000, 4500, 8000, 9000, 10000,
                         10800, 12000, 14390, 15000, 18010,
                         20000, 23000, 25000, 27800,
                         30000, 33000, 35000, 37000,
                         40000, 45000, 47040,
                         50000, 55020,
                         60000, 65000, 67000,
                         70000, 75000,
                         80000, 85000,
                         90000, 95000,
                         100000, 101300, 140020,
                         200000, 300000, 400000, 500000, 1166000])


if __name__ == "__main__":

    eps = 100000
    data_size_str_list = ['2000', '4000', '8000', '16000',  '32000']
    k = 3
    SEEDS = [1]
    QUANTILES = 30
    DATA = [
        ('folktables_2018_coverage_CA', 'PUBCOV'),
        ('folktables_2018_mobility_CA', 'MIG'),
        ('folktables_2018_employment_CA', 'ESR'),
        ('folktables_2018_income_CA', 'PINCP'),
        ('folktables_2018_travel_CA', 'JWMNP'),
    ]

    for (dataset_name, target), data_size_str in itertools.product(DATA, data_size_str_list):
        for seed in SEEDS:
            key = jax.random.PRNGKey(seed)

            root_path = 'dp-data-dev/datasets/preprocessed/folktables/1-Year/'
            config = load_domain_config(dataset_name, root_path=root_path)
            df_train = load_df(dataset_name, root_path=root_path, idxs_path=f'seed{seed}/train')
            df_test = load_df(dataset_name, root_path=root_path, idxs_path=f'seed{seed}/test')


            bins_edges = {}

            quantiles = np.linspace(0, 1, QUANTILES)
            for att in config:
                if config[att]['type'] == 'numerical':
                    v = df_train[att].values
                    thresholds = np.quantile(v, q=quantiles)
                    bins_edges[att] = thresholds

            domain = Domain(config=config, bin_edges=bins_edges)
            data = Dataset(df_train, domain)

            N = len(data.df)

            data_size = N if data_size_str == 'N' else int(data_size_str)

            print(f'Input data {dataset_name}, epsilon={eps:.2f}, data_size={data_size}, k={k} ')

            # Create statistics and evaluate
            key = jax.random.PRNGKey(0)
            # One-shot queries
            modules = []

            modules.append(Marginals.get_all_kway_combinations(data.domain, k=k, levels=1, bin_edges=bins_edges, include_feature=target))
            stat_module = ChainedStatistics(modules)
            stat_module.fit(data, max_queries_per_workload=30000)

            true_stats = stat_module.get_all_true_statistics()
            stat_fn0 = stat_module._get_workload_fn()

            N = len(data.df)
            algo = GSD(num_generations=600000,
                               print_progress=False,
                               stop_early=True,
                               domain=data.domain,
                               population_size=50,
                               data_size=data_size,
                               stop_early_gen=data_size,
                               )
            # Choose algorithm parameters

            # delta = 1.0 / len(data) ** 2
            delta = 10**(-5)
            # Generate differentially private synthetic data with ADAPTIVE mechanism

            t0 = timer()

            sync_data = algo.fit_dp(key, stat_module=stat_module,
                               epsilon=eps,
                               delta=delta)

            sync_dir = f'sync_data/{dataset_name}/{k}/{eps:.2f}/{data_size_str}/oneshot'
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