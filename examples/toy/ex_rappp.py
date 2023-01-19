import itertools
import os

import jax
from models import Generator, RelaxedProjectionPP
from stats import Halfspace
from stats.halfspaces_v3 import Halfspace3
from toy_datasets.circles import get_circles_dataset
from toy_datasets.moons import get_moons_dataset
import matplotlib.pyplot as plt
import time
from utils.plot_low_dim_data import plot_2d_data


def run_toy_example(data, debug_fn):
    rounds = 40
    epsilon = 1.0
    seed = 0
    data_size = 500

    plot_2d_data(data.to_numpy())
    debug_fn(0, data)
    key_hs = jax.random.PRNGKey(0)
    stats_module, kway_combinations = Halfspace3.get_kway_random_halfspaces(data.domain, k=1, rng=key_hs,
                                                                           random_hs=1000)
    stats_module.fit(data)
    print(f'workloads = ', len(stats_module.true_stats))

    rappp = RelaxedProjectionPP(
                        domain=data.domain,
                        data_size=data_size,
                    learning_rate=(0.0001, ),
                    print_progress=False)

    print(f'Starting {rappp}:')
    stime = time.time()
    key = jax.random.PRNGKey(seed)

    sync_data = rappp.fit_dp_adaptive(key, stat_module=stats_module,  epsilon=epsilon, delta=1e-6,
                                        rounds=rounds, print_progress=True, debug_fn=debug_fn)

    errors = jax.numpy.abs(stats_module.get_true_stats() - stats_module.get_stats_jit(sync_data))
    ave_error = jax.numpy.linalg.norm(errors, ord=1)
    print(f'{str(rappp)}: max error = {errors.max():.4f}, ave_error={ave_error:.6f}, time={time.time()-stime:.4f}')


DATASETS = {
    # 'circles': get_circles_dataset(DATA_SIZE=10000),
            'moons': get_moons_dataset(DATA_SIZE=10000)
}

if __name__ == "__main__":

    for data_name in DATASETS:
        data = DATASETS[data_name]
        os.makedirs(f'images/rappp/{data_name}/', exist_ok=True)
        def debug_fn(t, tempdata):
            if t == 0:
                save_path = f'images/rappp/{data_name}_original.png'
            else:
                save_path = f'images/rappp/{data_name}/img_{t:04}.png'
                
            plot_2d_data(tempdata.to_numpy(), title=f'epoch={t}', alpha=0.9, save_path=save_path)
        run_toy_example(data, debug_fn)