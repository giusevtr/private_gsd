import itertools

import folktables
import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSEmployment
from utils import Dataset, Domain, DataTransformer
from models_v2 import Generator, PrivGA, RelaxedProjectionPP
from stats_v2 import Marginals
from utils.utils_data import get_data
import sys, os
import jax


"""
Runtime analysis of PrivGA
Split the ACS datasets. 
RAP datasize = 50000
3-way marginals
3 random seeds
T= [25, 50, 100]
"""
ALGORITHMS = [
    PrivGA(popsize=100,
            top_k=50,
            num_generations=2000,
            stop_loss_time_window=100,
            print_progress=True,
            start_mutations=256,
            data_size=1000)
]

def run_experiments(epsilon=(0.07, 0.15, 0.23, 0.41, 0.52, 0.62, 0.74, 0.87, 1.0)):

    tasks = ['employment', 'coverage', 'income', 'mobility', 'travel']
    states = ['NY', 'CA',  'FL', 'TX', 'PA']

    for task, state in itertools.product(tasks, states):
        data_name =f'folktables_{task}_2018_{state}'
        data = get_data(f'folktables_datasets/{data_name}-mixed',
                 domain_name=f'folktables_datasets/{data_name}-mixed')

        # stats_module = TwoWayPrefix.get_stat_module(data.domain, num_rand_queries=1000000)
        stats_module = Marginals.get_all_kway_combinations(data.domain, k=3)
        stats_module.fit(data)

        for T, eps, seed in itertools.product([25, 50, 100], list(epsilon), [0, 1, 2]):

            for algorithm in ALGORITHMS:
                algorithm: Generator
                key = jax.random.PRNGKey(seed)
                sync_data_2 = algorithm.fit_dp_adaptive(key, stat_module=stats_module,
                                                      rounds=T, epsilon=eps, delta=1e-6, print_progress=True)
                errros = stats_module.get_sync_data_errors(sync_data_2.to_numpy())

                print(f'{str(algorithm)}: max error = {errros.max():.5f}')

                algo_name = str(algorithm)
                save_path = 'sync_datasets'
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, data_name)
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, algo_name)
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, f'{T:03}')
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, f'{eps:.2f}')
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, f'sync_data_{seed}.csv')
                data_df: pd.DataFrame = sync_data_2.df
                print(f'Saving {save_path}')
                data_df.to_csv(save_path)




if __name__ == "__main__":
    # df = folktables.
    run_experiments()
