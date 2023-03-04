import os
import jax.random
import pandas as pd

from models import GeneticSD, GeneticStrategy
from stats import ChainedStatistics,  Marginals, NullCounts
import jax.numpy as jnp
from dp_data import load_domain_config, load_df
from utils import timer, Dataset, Domain, filter_outliers
import pickle


if __name__ == "__main__":
    # epsilon_vals = [10]
    epsilon_vals = [10]

    dataset_name = 'national2019'
    root_path = '../../../dp-data-dev/datasets/preprocessed/sdnist_dce/'
    config = load_domain_config(dataset_name, root_path=root_path)
    df_train = load_df(dataset_name, root_path=root_path)

    domain = Domain(config)
    data = Dataset(df_train, domain)

    # Create statistics and evaluate
    key = jax.random.PRNGKey(0)
    module0 = Marginals.get_all_kway_combinations(data.domain, k=2, bins=[2, 4, 8])
    module1 = Marginals.get_all_kway_combinations(data.domain, k=1, bins=[100, 200, 400])
    module2 = NullCounts(domain)

    stat_module = ChainedStatistics([module0, module1, module2])
    stat_module.fit(data)

    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module._get_workload_fn()

    algo = GeneticSD(num_generations=50000, print_progress=True, stop_early=True, strategy=GeneticStrategy(domain=data.domain, elite_size=2, data_size=2000))
    # Choose algorithm parameters

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

            # post_sync_data = preprocessor.inverse_transform(sync_data.df)
            sync_data.df.to_csv(f'{sync_dir}/sync_data_{seed}.csv', index=False)
            errors = jnp.abs(true_stats - stat_fn(sync_data.to_numpy()))
            print(f'GSD(oneshot): eps={eps:.2f}, seed={seed}'
                  f'\t max error = {errors.max():.5f}'
                  f'\t avg error = {errors.mean():.5f}'
                  f'\t time = {timer() - t0:.4f}')

        print()

