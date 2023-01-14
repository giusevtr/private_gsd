import jax.random
import jax.numpy as jnp
from models import PrivGA, SimpleGAforSyncData
from stats import Marginals
from utils.utils_data import get_data
from utils.utils_data import Dataset
import numpy as np

import time


if __name__ == "__main__":

    # Get Data
    ROUNDS = 4
    # adaptive_rounds = (3, 10, 100)

    task = 'mobility'
    state = 'CA'
    data_name = f'folktables_2018_{task}_{state}'
    data = get_data(f'folktables_datasets/{data_name}-mixed-train',
                    domain_name=f'folktables_datasets/domain/{data_name}-num',  root_path='../../data_files/')

    # Real values must be in [0,1]

    # Create statistics and evaluate
    BINS = [2, 4, 8, 16, 32]
    # marginal_module, kway = Marginals.get_all_kway_mixed_combinations_v1(data.domain, k=2, bins=BINS)
    marginal_module, kway = Marginals.get_all_kway_mixed_combinations(data.domain, k_disc=0, k_real=2, bins=BINS)
    marginal_module.fit(data)

    print(f'Workloads = {len(marginal_module.true_stats)}')

    ########
    # PrivGA
    ########
    data_size = 2000
    priv_ga = PrivGA(
        num_generations=100000,
        stop_loss_time_window=50,
        print_progress=False,
        strategy=SimpleGAforSyncData(
            domain=data.domain,
            data_size=data_size,
            population_size=100,
            elite_size=10,
            muta_rate=1,
            mate_rate=40
        )
    )

    # Generate differentially private synthetic data with ADAPTIVE mechanism
    key = jax.random.PRNGKey(0)
    stime = time.time()

    sync_data = priv_ga.fit_dp_adaptive(key, stat_module=marginal_module, rounds=ROUNDS,
                                 epsilon=1, delta=1e-6, tolerance=0.0, print_progress=True)

    true_stats = marginal_module.get_true_stats()
    sync_stats = marginal_module.get_stats_jit(sync_data)
    print(f'PrivGA: max error = {jnp.abs(true_stats - sync_stats).max():.5f}, '
          f'ave error = {jnp.linalg.norm(true_stats - sync_stats, ord=1) / true_stats.shape[0]:.7f}\t'
          f'time = {time.time() - stime:.5f}')

