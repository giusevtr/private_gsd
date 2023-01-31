import itertools
import os
import jax
from models import PrivGAfast, SimpleGAforSyncDataFast, RelaxedProjectionPP
from stats import Halfspace, Prefix, Marginals
import time
from utils.utils_data import get_data
import pandas as pd
import argparse


def run_acs_example(generator,
                    data,
                    data_name,
                    stats_module,
                    epsilon=1.00,
                    seed=0,
                    rounds=30,
                    num_sample=10,
                    adaptive=False):
    print(f'Running {generator} with epsilon={epsilon}, train module is {stats_module}')
    if adaptive:
        print(f'Adaptive with {rounds} rounds and {num_sample} samples.')
        folder = f'sync_data/{data_name}/{str(generator)}/{str(stats_module)}/{rounds}/{num_sample}/{epsilon:.2f}'
        print(folder)
        os.makedirs(folder, exist_ok=True)
    else:
        print('Non-adaptive')
        folder = f'sync_data/{data_name}/{str(generator)}/{str(stats_module)}/non-adaptive/{epsilon:.2f}'
        os.makedirs(folder, exist_ok=True)
    stats_module.fit(data)
    path = f'{folder}/sync_data_{seed}.csv'

    print(f'Starting {generator}:')
    stime = time.time()
    key = jax.random.PRNGKey(seed)
    delta = 1 / len(data) ** 2
    if adaptive:
        sync_data = generator.fit_dp_adaptive(key, stat_module=stats_module, epsilon=epsilon, delta=delta,
                                              rounds=rounds, print_progress=False, num_sample=num_sample)
    else:
        sync_data = generator.fit_dp(key, stat_module=stats_module, epsilon=epsilon, delta=delta)
    etime = time.time() - stime
    print(f'Saving in {path}')
    sync_data.df.to_csv(path, index=False)

    errors = jax.numpy.abs(stats_module.get_true_stats() - stats_module.get_stats_jit(sync_data))
    ave_error = jax.numpy.linalg.norm(errors, ord=1) / errors.shape[0]
    print(
        f'{str(generator)}: Train max error = {errors.max():.4f}, ave_error={ave_error:.6f}, time={time.time() - stime:.4f}')
    return errors.max(), ave_error, etime
