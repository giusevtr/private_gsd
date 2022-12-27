import itertools

import jax.random

from examples.run_example import generate_private_SD, run_experiment
from models_v3 import PrivGA, SimpleGAforSyncData, RelaxedProjection
from stats_v3 import Marginals
from utils.utils_data import get_data
import time
import pdb

if __name__ == "__main__":

    # Get Data
    ROUNDS = 15
    data = get_data('adult', 'adult-small', root_path='../../data_files/')
    # ROUNDS=15
    # data = get_data('adult', 'adult', root_path='../data_files/')

    # Create statistics and evaluate
    marginal_module = Marginals.get_all_kway_combinations(data.domain, k=2)
    marginal_module.fit(data)


    # stime = time.time()
    # rap = RelaxedProjection(domain=data.domain, data_size=1000, iterations=1000, learning_rate=0.05, print_progress=False)
    # key = jax.random.PRNGKey(0)
    # sync_data_rap = rap.fit_dp_adaptive(key, stat_module=marginal_module, rounds=ROUNDS,
    #                                       epsilon=0.01, delta=1e-6, tolerance=0.01, print_progress=True)
    # errros = marginal_module.get_sync_data_errors(sync_data_rap.to_numpy())
    # print(f'RAP: max error = {errros.max():.5f}, time={time.time() - stime:.4f}s\n')



    data_size = 5000
    strategy = SimpleGAforSyncData(
            domain=data.domain,
            data_size=data_size,
            population_size=5000,
            elite_size=10,
            mute_rate=100,
            mate_rate=500
        )

    # for cross, mut in itertools.product([0.01, 0.05, 0.1, 0.2], [1, 10, 100, 1000]):
    #     print(f'C={cross:.2f}, mut={mut:<4}')
    # Choose algorithm parameters
    priv_ga = PrivGA(
                    domain=data.domain,
                    data_size=data_size,
                    num_generations=2000,
                    stop_loss_time_window=50,
                    print_progress=False,
                    # start_mutations=100,
                    strategy=strategy
                     )
    stime = time.time()
    # key = jax.random.PRNGKey(0)
    key = jax.random.PRNGKey(0)
    sync_data_2 = priv_ga.fit_dp_adaptive(key, stat_module=marginal_module, rounds=ROUNDS,
                                 epsilon=0.01, delta=1e-6, tolerance=0.00, start_X=False, print_progress=True)
    errros = marginal_module.get_sync_data_errors(sync_data_2.to_numpy())
    print(f'PrivGA: max error = {errros.max():.5f}, time={time.time() - stime:.4f}s\n')

