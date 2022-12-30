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

    data_size = 500
    data = get_data('adult', 'adult-small', root_path='../../data_files/')
    # ROUNDS=15
    # data = get_data('adult', 'adult', root_path='../data_files/')

    # Create statistics and evaluate
    marginal_module = Marginals.get_all_kway_combinations(data.domain, k=2)
    marginal_module.fit(data)
    ROUNDS = 10
    rap = RelaxedProjection(domain=data.domain, data_size=1000, iterations=1000, learning_rate=0.05, print_progress=False)

    for SEED in [0, 1, 2]:
        # Get Data


        #####
        # RAP
        #####
        stime = time.time()
        key = jax.random.PRNGKey(SEED)
        sync_data_rap = rap.fit_dp_adaptive(key, stat_module=marginal_module, rounds=ROUNDS,
                                              epsilon=0.01, delta=1e-6, tolerance=0.0, start_X=True, print_progress=True)
        errros = marginal_module.get_sync_data_errors(sync_data_rap.to_numpy())
        print(f'RAP: max error = {errros.max():.5f}, time={time.time() - stime:.4f}s\n')

        rap.ADA_DATA.to_csv(f'res/rap_results_{SEED}.csv', index=False)