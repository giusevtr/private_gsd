import itertools
import os

import jax
import jax.numpy as jnp
from models import Generator, PrivGAfast, SimpleGAforSyncDataFast, RelaxedProjectionPP
from stats import Halfspace, Prefix
from stats.halfspaces import Halfspace, Marginals
from toy_datasets.circles import get_circles_dataset
from toy_datasets.moons import get_moons_dataset
from toy_datasets.sparse import get_sparse_dataset
import matplotlib.pyplot as plt
import time
from utils.plot_low_dim_data import plot_2d_data
from utils.utils_data import get_data
import argparse


def run_acs_example(algo,
                    data,
                    stats_module,
                    epsilon=1.00,
                    seed=0,
                    rounds=30,
                    num_sample=10,
                    adaptive=False):
    print(f'Running {algo} with epsilon={epsilon}, train module is {stats_module}')
    if adaptive:
        print(f'Adaptive with {rounds} rounds and {num_sample} samples.')
    else:
        print('Non-adaptive')
    stats_module.fit(data)
    data_name = f'folktables_2018_real_CA'
    folder = f'sync_data/{str(algo)}/{stats_module}/{data_name}/{epsilon:.2f}/{rounds}'
    os.makedirs(folder, exist_ok=True)
    path = f'{folder}/sync_data_{seed}.csv'


    print(f'Starting {algo}:')
    stime = time.time()
    key = jax.random.PRNGKey(seed)

    if adaptive:
        sync_data = algo.fit_dp_adaptive(key, stat_module=stats_module,  epsilon=epsilon, delta=1e-6,
                                        rounds=rounds, print_progress=True, num_sample=num_sample)
    else:
        sync_data = algo.fit_dp(key, stat_module=stats_module,  epsilon=epsilon, delta=1e-6)

    print(f'Saving in {path}')
    sync_data.df.to_csv(path, index=False)

    errors = jax.numpy.abs(stats_module.get_true_stats() - stats_module.get_stats_jit(sync_data))
    ave_error = jax.numpy.linalg.norm(errors, ord=1)/errors.shape[0]
    print(f'{str(algo)}: Train max error = {errors.max():.4f}, ave_error={ave_error:.6f}, time={time.time()-stime:.4f}')


if __name__ == "__main__":

    data_name = f'folktables_2018_real_CA'
    data = get_data(f'{data_name}-mixed-train',
                    domain_name=f'domain/{data_name}-mixed',  root_path='../../data_files/folktables_datasets_real')


    # algo = PrivGAfast(num_generations=100000, print_progress=False, strategy=SimpleGAforSyncDataFast(
    #         domain=data.domain, data_size=2000, population_size=100, elite_size=5, muta_rate=1, mate_rate=1))
    algo = RelaxedProjectionPP(domain=data.domain, data_size=1000, learning_rate=(0.01,), print_progress=False)


    train_module, _ = Halfspace.get_kway_random_halfspaces(data.domain, k=1, rng=jax.random.PRNGKey(0), random_hs=20000)
    # eval_module, _ = Halfspace4.get_kway_random_halfspaces(data.domain, k=1, rng=jax.random.PRNGKey(1), random_hs=2000, )
    # ranges_stat_module, _ = Marginals.get_all_kway_mixed_combinations(data.domain, k_disc=1, k_real=2,
    #                                                                   bins=[2, 4, 8, 16, 32, 64])
    # train_module = Prefix.get_kway_prefixes(data.domain, k=1, rng=jax.random.PRNGKey(0), random_prefixes=20000)[0]

    for eps in [0.07, 0.23, 0.52]:
        for r in [50, 75]:
            for seed in [0]:

                run_acs_example(algo, data,
                                stats_module=train_module,
                                epsilon=eps,
                                seed=seed,
                                rounds=r,
                                num_sample=10,
                                adaptive=True)