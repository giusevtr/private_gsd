import itertools
import time

import folktables
import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSEmployment
from utils import Dataset, Domain, DataTransformer
from models_v2 import Generator, PrivGA, RelaxedProjectionPP
from stats_v2 import Marginals
from utils.utils_data import get_data
import os
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
    PrivGA(popsize=popsize,
           top_k=top_k,
           num_generations=10000,
           stop_loss_time_window=100,
           print_progress=True,
           start_mutations=256,
           data_size=1000)
    for popsize, top_k in itertools.product([1000],[2])
]


def run_experiments(epsilon=(1.0,)):
    # tasks = ['employment', 'coverage', 'income', 'mobility', 'travel']
    # states = ['NY', 'CA', 'FL', 'TX', 'PA']
    tasks = ['employment']
    states = ['CA']
    RESULTS = []
    for task, state in itertools.product(tasks, states):
        data_name = f'folktables_2018_{task}_{state}'
        data = get_data(f'folktables_datasets/{data_name}-mixed-train',
                        domain_name=f'folktables_datasets/domain/{data_name}-cat')

        # stats_module = TwoWayPrefix.get_stat_module(data.domain, num_rand_queries=1000000)
        stats_module = Marginals.get_all_kway_combinations(data.domain, k=3)
        stats_module.fit(data)

        for eps, seed in itertools.product(list(epsilon), [1,2]):
            print(eps)
            for algorithm in ALGORITHMS:
                algorithm: Generator
                key = jax.random.PRNGKey(seed)
                stime = time.time()
                delta = 1.0 / len(data) ** 2
                sync_data_2,end_round = algorithm.fit_dp(key, stat_module=stats_module,
                                                        epsilon=eps, delta=delta)
                total_time = time.time()-stime
                print("total_time:",total_time,"end_round:",end_round)
                # errors = stats_module.get_sync_data_errors(sync_data_2.to_numpy())
                RESULTS.append(
                    [data_name, str(algorithm), algorithm.popsize, algorithm.top_k, str(stats_module), eps, seed, end_round, total_time])
                # print(f'{str(algorithm)}: max error = {errors.max():.5f}')

                # algo_name = str(algorithm)
                # save_path = 'sync_datasets'
                # os.makedirs(save_path, exist_ok=True)
                # save_path = os.path.join(save_path, data_name)
                # os.makedirs(save_path, exist_ok=True)
                # save_path = os.path.join(save_path, algo_name)
                # os.makedirs(save_path, exist_ok=True)
                # save_path = os.path.join(save_path, f'{T:03}')
                # os.makedirs(save_path, exist_ok=True)
                # save_path = os.path.join(save_path, f'{eps:.2f}')
                # os.makedirs(save_path, exist_ok=True)
                # save_path = os.path.join(save_path, f'sync_data_{seed}.csv')
                # data_df: pd.DataFrame = sync_data_2.df
                # print(f'Saving {save_path}')
                # data_df.to_csv(save_path)
            results_df = pd.DataFrame(RESULTS,
                                      columns=['data', 'generator','popsize','top_k', 'stats', 'epsilon', 'seed', 'round', 'time'])
            results_df.to_csv("parameter1.csv")
if __name__ == "__main__":
    # df = folktables.
    run_experiments()
