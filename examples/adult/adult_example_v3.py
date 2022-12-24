import jax.random

from examples.run_example import generate_private_SD, run_experiment
from models_v3 import PrivGA, SimpleGAforSyncData
from stats_v3 import Marginals
from utils.utils_data import get_data
import time
import pdb

if __name__ == "__main__":

    # Get Data
    ROUNDS = 7
    data = get_data('adult', 'adult-small', root_path='../../data_files/')
    # ROUNDS=15
    # data = get_data('adult', 'adult', root_path='../data_files/')

    # Create statistics and evaluate
    marginal_module = Marginals.get_all_kway_combinations(data.domain, k=2)
    marginal_module.fit(data)
    strategy = SimpleGAforSyncData(
        domain=data.domain,
        # data_size=500,
        # popsize=1000,
        # elite_popsize=2,
    )

    # Choose algorithm parameters
    priv_ga = PrivGA(
                    domain=data.domain,
                    data_size=500,
                    num_generations=1000,
                    popsize=1000,
                    elite_popsize=30,
                    stop_loss_time_window=20,
                    print_progress=True,
                    start_mutations=32,
                    cross_rate=0.1,
                    strategy=strategy
                     )
    for i in range(2):
        stime = time.time()
        key = jax.random.PRNGKey(i)
        sync_data_2 = priv_ga.fit_dp_adaptive(key, stat_module=marginal_module, rounds=ROUNDS,
                                     epsilon=1, delta=1e-6, tolerance=0.01, print_progress=True)
        errros = marginal_module.get_sync_data_errors(sync_data_2.to_numpy())
        print(f'Run {i}) PrivGA: max error = {errros.max():.5f}, time={time.time() - stime:.4f}s\n')

