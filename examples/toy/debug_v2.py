import itertools
import os

import jax
from models import PrivGA, SimpleGAforSyncData
from stats import Halfspace, Marginals
from toy_datasets.circles import get_circles_dataset
from toy_datasets.moons import get_moons_dataset
from toy_datasets.classification import get_classification
import matplotlib.pyplot as plt
import time
from utils.plot_low_dim_data import plot_2d_data


if __name__ == "__main__":

    data = get_classification(DATA_SIZE=1000, d=3, seed=9)
    bins = [2, 4, 8, 16, 32]
    stats_module, kway_combinations = Marginals.get_all_kway_combinations(data.domain, k=2, bins=bins)
    stats_module.fit(data)
    priv_ga = PrivGA(
                    num_generations=100000,
                    strategy=SimpleGAforSyncData(
                        domain=data.domain,
                        data_size=100,
                        population_size=100,
                        elite_size=5,
                        muta_rate=1,
                        mate_rate=1,
                    ),
                    print_progress=True)
    key = jax.random.PRNGKey(0)
    delta = 1 / len(data.df_real) ** 2
    sync_data = priv_ga.fit_dp_adaptive(key, stat_module=stats_module, epsilon=1, delta=delta, rounds=3, print_progress=True)
    # sync_data = priv_ga.fit_dp(key, stat_module=stats_module, epsilon=1, delta=delta)

    all_stats_fn = stats_module.get_all_statistics_fn()
    errors = jax.numpy.abs(stats_module.get_all_true_statistics() - all_stats_fn(sync_data.to_numpy()))
    ave_error = jax.numpy.linalg.norm(errors, ord=1) / errors.shape[0]
    print(f'{str(priv_ga)}: Train max error = {errors.max():.4f}, ave_error={ave_error:.6f}')
