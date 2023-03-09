import os
import jax.random
import pandas as pd

from models import GeneticSD, GeneticStrategy
from stats import ChainedStatistics,  Marginals, NullCounts, Prefix
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
    # One-shot queries
    module0 = Marginals.get_all_kway_combinations(data.domain, k=2, bins=[2, 4, 8, 16, 32, 64])


    # Adaptive queries
    module1 = Prefix.get_kway_prefixes(data.domain, k_cat=1, k_num=2, random_prefixes=50000, rng=key)
    module3 = Marginals.get_all_kway_combinations(data.domain, k=3, bins=[2, 4, 8, 16, 32, 64])
    # module0 = Marginals.get_all_kway_combinations(data.domain, k=2, bins=[50, 100, 200])
    # module1 = Marginals.get_all_kway_combinations(data.domain, k=1, bins=[2500, 5000, 10000])

    module2 = NullCounts(domain)

    stat_module = ChainedStatistics([module0, module1, module2, module3])
    stat_module.fit(data)

    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module._get_workload_fn()

    algo = GeneticSD(num_generations=50000, print_progress=True, stop_early=True, strategy=GeneticStrategy(domain=data.domain, elite_size=2, data_size=1000))
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


            sync_data = algo.fit_dp_hybrid(key, stat_module=stat_module, oneshot_stats_ids=[0],
                                oneshot_share=0.7,
                               rounds=20,
                               epsilon=eps,
                               delta=delta)
            # sync_data = algo.fit_dp(key, stat_module=stat_module, epsilon=eps, delta=delta)

            # post_sync_data = preprocessor.inverse_transform(sync_data.df)
            sync_data.df.to_csv(f'{sync_dir}/sync_data_{seed}.csv', index=False)
            errors = jnp.abs(true_stats - stat_fn(sync_data.to_numpy()))
            print(f'GSD(oneshot): eps={eps:.2f}, seed={seed}'
                  f'\t max error = {errors.max():.5f}'
                  f'\t avg error = {errors.mean():.5f}'
                  f'\t time = {timer() - t0:.4f}')

        print()

