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

    eps = 100000
    data_size_str = '16000'
    k = 3
    SEEDS = [0]
    QUANTILES = 30
    DATA = [
        # 'folktables_2018_coverage_CA',
        # 'folktables_2018_mobility_CA',
        # 'folktables_2018_employment_CA',
        # 'folktables_2018_income_CA',
        'folktables_2018_travel_CA',
    ]

    for dataset_name in DATA:

        root_path = 'dp-data-dev/datasets/preprocessed/folktables/1-Year/'
        config = load_domain_config(dataset_name, root_path=root_path)
        df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
        df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')


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

        modules.append(Marginals.get_all_kway_combinations(data.domain, k=2, levels=1, bin_edges=bins_edges))
        stat_module = ChainedStatistics(modules)
        stat_module.fit(data, max_queries_per_workload=10000)

        true_stats = stat_module.get_all_true_statistics()
        stat_fn0 = stat_module._get_workload_fn()

        N = len(data.df)
        algo = GSD(num_generations=3000000,
                           print_progress=True,
                           stop_early=True,
                           domain=data.domain,
                           population_size=40,
                           data_size=data_size,
                           stop_early_gen=data_size,
                           )
        # Choose algorithm parameters

        # delta = 1.0 / len(data) ** 2
        delta = 10**(-5)
        # Generate differentially private synthetic data with ADAPTIVE mechanism
        for seed in SEEDS:
            key = jax.random.PRNGKey(seed)
            t0 = timer()

            sync_data = algo.fit_dp_adaptive(key, stat_module=stat_module,
                                             rounds=100,
                                             num_sample=1,
                               epsilon=eps,
                               delta=delta)

            sync_dir = f'sync_data/{dataset_name}/{k}/{eps:.2f}/{data_size_str}/100'
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