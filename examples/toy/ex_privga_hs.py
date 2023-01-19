import itertools
import os

import jax
import jax.numpy as jnp
from models import Generator, PrivGAfast, SimpleGAforSyncDataFast
from stats import Halfspace
from stats.halfspaces_v4 import Halfspace4, Marginals
from toy_datasets.circles import get_circles_dataset
from toy_datasets.moons import get_moons_dataset
import matplotlib.pyplot as plt
import time
from utils.plot_low_dim_data import plot_2d_data


def run_toy_example(data):
    epsilon = 1.00
    seed = 0
    data_size = 300
    random_hs = 5000
    rounds = 30
    adaptive = False
    print_progress = False

    # plot_2d_data(data.to_numpy())
    key_hs = jax.random.PRNGKey(0)
    stats_module, kway_combinations = Halfspace4.get_kway_random_halfspaces(data.domain, k=1, rng=key_hs,
                                                                           random_hs=random_hs)
    stats_module.fit(data)
    print(f'workloads = ', len(stats_module.true_stats))
    bins = [2, 4, 8, 16, 32]
    evaluate_stat_module, kway_combinations = Marginals.get_all_kway_mixed_combinations(data.domain, k_disc=1, k_real=2,
                                                                                bins=bins)
    evaluate_stat_module.fit(data)



    folder = f'images/privga/halfspaces/{data_name}/'
    os.makedirs(folder, exist_ok=True)
    def debug_fn(t, tempdata):
        X = tempdata.to_numpy()
        train_error = jnp.abs(stats_module.get_true_stats() - stats_module.get_stats_jax_jit(X)).max()
        eval_error = jnp.abs(evaluate_stat_module.get_true_stats() - evaluate_stat_module.get_stats_jax_jit(X)).max()
        if t == 0:
            save_path = f'{folder}/{data_name}_original.png'
        else:
            save_path = f'{folder}/img_{t:04}.png'

        plot_2d_data(tempdata.to_numpy(), title=f'epoch={t}:\nTrain error = {train_error:.3f}, Eval error={eval_error:.3f}', alpha=0.9, save_path=save_path)



    priv_ga = PrivGAfast(
                    num_generations=100000,
                    strategy=SimpleGAforSyncDataFast(
                        domain=data.domain,
                        data_size=data_size,
                        population_size=100,
                        elite_size=5,
                        muta_rate=1,
                        mate_rate=1,
                    ),
                    print_progress=print_progress)

    print(f'Starting {priv_ga}:')
    stime = time.time()
    key = jax.random.PRNGKey(seed)

    if adaptive:
        sync_data = priv_ga.fit_dp_adaptive(key, stat_module=stats_module,  epsilon=epsilon, delta=1e-6,
                                        rounds=rounds, print_progress=True, debug_fn=debug_fn)
    else:
        sync_data = priv_ga.fit_dp(key, stat_module=stats_module,  epsilon=epsilon, delta=1e-6)

    debug_fn(1, sync_data)
    errors = jax.numpy.abs(stats_module.get_true_stats() - stats_module.get_stats_jit(sync_data))
    ave_error = jax.numpy.linalg.norm(errors, ord=1)
    print(f'{str(priv_ga)}: Train max error = {errors.max():.4f}, ave_error={ave_error:.6f}, time={time.time()-stime:.4f}')


    eval_errors = jax.numpy.abs(evaluate_stat_module.get_true_stats() - evaluate_stat_module.get_stats_jit(sync_data))
    eval_ave_error = jax.numpy.linalg.norm(eval_errors, ord=1)
    print(f'{str(priv_ga)}:  Eval max error = {eval_errors.max():.4f}, ave_error={eval_ave_error:.6f}, time={time.time()-stime:.4f}')



DATASETS = {
    # 'circles': get_circles_dataset(DATA_SIZE=10000),
            'moons': get_moons_dataset(DATA_SIZE=10000)
}

if __name__ == "__main__":

    for data_name in DATASETS:
        data = DATASETS[data_name]

        run_toy_example(data)