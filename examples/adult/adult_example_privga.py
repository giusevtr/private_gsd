import itertools

import jax.random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from examples.run_example import generate_private_SD, run_experiment
from models_v3 import PrivGA, SimpleGAforSyncData, RelaxedProjection
from stats_v3 import Marginals
from utils.utils_data import get_data
import time
import pdb

if __name__ == "__main__":
    ROUNDS = 10

    # Get Data
    data = get_data('adult', 'adult-small', root_path='../../data_files/')
    # ROUNDS=15
    # data = get_data('adult', 'adult', root_path='../data_files/')

    # Create statistics and evaluate
    marginal_module = Marginals.get_all_kway_combinations(data.domain, k=2)
    marginal_module.fit(data)


    data_size = 100
    strategy = SimpleGAforSyncData(
            domain=data.domain,
            data_size=data_size,
            population_size=100,
            elite_size=5,
            muta_rate=1,
            mate_rate=10
        )

    ########
    # PrivGA
    ########
    priv_ga = PrivGA(
                    domain=data.domain,
                    data_size=data_size,
                    num_generations=10000,
                    stop_loss_time_window=20,
                    print_progress=False,
                    # start_mutations=100,
                    strategy=strategy
                     )
    for SEED in [0, 1, 2]:

        stime = time.time()
        key = jax.random.PRNGKey(SEED)
        sync_data_2 = priv_ga.fit_dp_adaptive(key, stat_module=marginal_module, rounds=ROUNDS,
                                     epsilon=0.01, delta=1e-6, tolerance=0.00, start_X=False, print_progress=True)
        errros = marginal_module.get_sync_data_errors(sync_data_2.to_numpy())
        print(f'PrivGA: max error = {errros.max():.5f}, time={time.time() - stime:.4f}s\n')
        priv_ga.ADA_DATA.to_csv(f'res/privga_results_{SEED}.csv', index=False)