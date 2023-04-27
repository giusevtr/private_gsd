import os

import jax.random
import matplotlib.pyplot as plt
import pandas as pd
from models import GeneticSD, GeneticStrategy, RelaxedProjectionPP
from stats import ChainedStatistics, Marginals, NullCounts
# from utils.utils_data import get_data
from utils import timer
import jax.numpy as jnp
# from dp_data.data import get_data
from dp_data import load_domain_config, load_df
from utils import timer, Dataset, Domain, filter_outliers
import seaborn as sns
import json


def get_constrains(domain: Domain):


    def constraint_fn():
        return 0

if __name__ == "__main__":
    # epsilon_vals = [0.07, 0.23, 0.52, 0.74, 1]
    epsilon_vals = [100]
    dataset_name = 'Nulldata'
    config = json.load(open('domain.json'))
    domain = Domain(config)

    data = Dataset.synthetic(domain, N=1000, seed=0, null_values=0.2)

    targets = ['PINCP',  'PUBCOV', 'ESR']

    # Create statistics and evaluate
    key = jax.random.PRNGKey(0)
    module0 = Marginals.get_all_kway_combinations(data.domain, k=2, bin_edges=[2, 4, 8, 16, 32])
    module1 = NullCounts(domain)
    stat_module = ChainedStatistics([module0, module1])
    stat_module.fit(data)

    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module._get_workload_fn()

    null_count_fn = module1._get_dataset_statistics_fn()
    algo = GeneticSD(num_generations=5000, print_progress=True, stop_early=True, strategy=GeneticStrategy(domain=data.domain, elite_size=2, data_size=1000))
    # Choose algorithm parameters

    print(f'Nulls in original data: {null_count_fn(data)}')
    delta = 1.0 / len(data) ** 2
    # Generate differentially private synthetic data with ADAPTIVE mechanism
    # for eps in [1]:
    for eps in epsilon_vals:
        for seed in [0]:
        # for seed in [0]:
            sync_dir = f'sync_data/{dataset_name}/GSD/Ranges/oneshot/{eps:.2f}/'
            os.makedirs(sync_dir, exist_ok=True)

            key = jax.random.PRNGKey(seed)
            t0 = timer()

            sync_data = algo.fit_dp(key, stat_module=stat_module, epsilon=eps, delta=delta)

            sync_data.df.to_csv(f'{sync_dir}/sync_data_{seed}.csv', index=False)
            errors = jnp.abs(true_stats - stat_fn(sync_data.to_numpy()))
            print(f'GSD(oneshot): eps={eps:.2f}, seed={seed}'
                  f'\t max error = {errors.max():.5f}'
                  f'\t avg error = {errors.mean():.5f}'
                  f'\t time = {timer() - t0:.4f}')
            print('Final ML Results:')

            print(f'Nulls in synthetic data: {null_count_fn(sync_data)}')
            print()

        print()

