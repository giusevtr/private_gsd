import itertools

import jax
import pandas as pd

from models import PrivGA
from stats import Marginals, ChainedStatistics
from dev.toy_datasets.sparsecat import get_sparsecat
import time
import jax.numpy as jnp


PRINT_PROGRESS = True
ROUNDS = 2
EPSILON = [1.5]
# EPSILON = [1]
SEEDS = [0]


def run(data, algo, marginal_module, eps=0.5, seed=0):
    print(f'Running {algo} with {marginal_module}')

    stat_module = ChainedStatistics([
                                         marginal_module,
                                         ])
    stat_module.fit(data)
    true_stats = stat_module.get_all_true_statistics()
    stat_fn = marginal_module._get_dataset_statistics_fn()


    stime = time.time()

    key = jax.random.PRNGKey(seed)
    sync_data = algo.fit_dp(key, stat_module=stat_module, epsilon=eps, delta=1/len(data.df)**2)
    # sync_data = algo.fit_dp_adaptive(key, stat_module=stat_module, epsilon=eps, delta=1/len(data.df)**2, rounds=3, print_progress=True)

    errors = jnp.abs(true_stats - stat_fn(sync_data))
    print(f'{algo}: eps={eps:.2f}, seed={seed}'
          f'\t max error = {errors.max():.5f}'
          f'\t avg error = {jnp.linalg.norm(errors, ord=2):.5f}')

    sel_true_stats = stat_module.get_selected_statistics_without_noise()
    noisy_stats = stat_module.get_selected_noised_statistics()
    gau_errors = jnp.abs(sel_true_stats - noisy_stats)
    print(f'\t Gau max error = {gau_errors.max():.5f}')
    print(f'\t Gau ave error = {jnp.linalg.norm(gau_errors, ord=2):.5f}')


if __name__ == "__main__":

    data = get_sparsecat(DATA_SIZE=2000, CAT_SIZE=1000)


    # stats = MarginalsDiff.get_all_kway_combinations(data.domain, k=2)
    # algo = RelaxedProjection(data.domain, 1000, 2000, learning_rate=0.01, print_progress=True)
    # run(data, algo, stats, seed=1)




    # population_size_muta = 200
    # population_size_cross = 0

    population_size_muta = 150
    population_size_cross = 50

    T = [
        # (200, 0),
        # (150, 50),
        # (100, 100),
        # (50, 150),
        (190, 10),
        (180, 20),
        (170, 30),
        (160, 40),
    ]

    df_all = []
    for population_size_muta,  population_size_cross in T:
        marginal_module = Marginals.get_all_kway_combinations(data.domain, k=1)
        priv_ga = PrivGA(num_generations=100000, print_progress=True,
                            domain=data.domain,
                            data_size=2000,
                            muta_rate=1,
                            mate_rate=1,
                            population_size_muta=population_size_muta,
                            population_size_cross=population_size_cross,
                          )

        print(f'population_size_muta={population_size_muta}, population_size_cross={population_size_cross}')
        run(data, priv_ga, marginal_module, eps=5, seed=1)

        fitness_df = pd.DataFrame(priv_ga.fitness_record, columns=['G', 'Fitness', 'Time'])
        fitness_df['Mutations'] = population_size_muta
        fitness_df['Crossovers'] = population_size_cross

        df_all.append(fitness_df)

    df_all = pd.concat(df_all, ignore_index=True)
    df_all.to_csv('fitness_progress_v2.csv')

