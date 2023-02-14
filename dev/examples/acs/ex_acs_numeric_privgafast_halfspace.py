import itertools

import jax.random
import jax.numpy as jnp
from models import PrivGA, SimpleGAforSyncData
from stats import Halfspace
from utils.utils_data import get_data
from utils.utils_data import Dataset

import time


if __name__ == "__main__":

    EPSILON = [0.07, 0.23, 0.52, 0.74, 1.00]
    # EPSILON = [ 1.00]
    # Get Data
    ROUNDS = 25
    # adaptive_rounds = (3, 10, 100)

    task = 'real'
    state = 'CA'
    data_name = f'folktables_2018_{task}_{state}'
    data = get_data(f'folktables_datasets/{data_name}-mixed-train',
                    domain_name=f'folktables_datasets/domain/{data_name}-mixed', root_path='../../../data_files/')
    # Create statistics and evaluate
    key = jax.random.PRNGKey(0)
    hs_module, kway = Halfspace.get_kway_random_halfspaces(data.domain, k=1, random_hs=10000, rng=key)
    hs_module.fit(data)

    print(f'Workloads = {len(hs_module.true_stats)}')

    ########
    # PrivGA
    ########
    data_size = 2000
    priv_ga = PrivGA(
        num_generations=100000,
        print_progress=False,
        strategy=SimpleGAforSyncData(
            domain=data.domain,
            data_size=data_size,
            population_size=1000,
            elite_size=2,
            muta_rate=1,
            mate_rate=1
        )
    )

    # for eps, rounds, seed in itertools.product(EPSILON, ROUNDS, [0]):
    #     # Generate differentially private synthetic data with ADAPTIVE mechanism
    #     key = jax.random.PRNGKey(seed)
    #     stime = time.time()
    #
    #     sync_data = priv_ga.fit_dp_adaptive(key, stat_module=hs_module, rounds=rounds, start_sync=True,
    #                                         epsilon=eps, delta=1e-6, print_progress=True,num_sample=1000)
    #
    #     true_stats = hs_module.get_true_stats()
    #     sync_stats = hs_module.get_stats_jit(sync_data)
    #     print(f'PrivGA(fast): eps={eps}, round={rounds}, seed={seed}'
    #           f'max error = {jnp.abs(true_stats - sync_stats).max():.5f}, '
    #           f'ave error = {jnp.linalg.norm(true_stats - sync_stats, ord=1) / true_stats.shape[0]:.7f}\t'
    #           f'time = {time.time() - stime:.5f}')
    sync_data = priv_ga.fit_dp_adaptive(key, stat_module=hs_module, rounds=ROUNDS, start_sync=True,
                                        epsilon=1.00, delta=1e-6, print_progress=True)
        # sync_data.df.to_csv('sync')


