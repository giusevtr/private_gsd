import itertools
import folktables
import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSEmployment
from utils import Dataset, Domain, DataTransformer
from models import Generator, PrivGA, SimpleGAforSyncData, RelaxedProjection
from stats import Marginals
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

tasks = ['mobility']
states = ['CA']
# tasks = ['employment', 'coverage', 'income', 'mobility', 'travel']
# states = ['NY', 'CA', 'FL', 'TX', 'PA']
EPSILON = (0.07, 0.23, 0.52, 0.74, 1.0)

def run_experiments(data: Dataset,
                    algorithm: Generator,
                    stats_module: Marginals,
                    epsilon=(0.07, 0.15, 0.23, 0.41, 0.52, 0.62, 0.74, 0.87, 1.0),
                    save_dir: tuple = ('sync_pat',)):


    N = data.df.shape[0]
    for T, eps, seed in itertools.product([25, 50, 100], list(epsilon), [0, 1, 2]):

        delta = 1/N**2
        key = jax.random.PRNGKey(seed)
        sync_data_2 = algorithm.fit_dp_adaptive(key, stat_module=stats_module,
                                                rounds=T, epsilon=eps, delta=delta, print_progress=True)
        errros = stats_module.get_sync_data_errors(sync_data_2.to_numpy())

        print(f'{str(algorithm)}: max error = {errros.max():.5f}')

        save_dir = list(save_dir)
        save_path = ''
        # save_path = save_dir
        for s in save_dir:
            save_path = os.path.join(save_path, s)
            os.makedirs(s, exist_ok=True)
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

    tasks = ['employment', 'coverage', 'income', 'mobility', 'travel']
    states = ['NY', 'CA', 'FL', 'TX', 'PA']

    for task, state in itertools.product(tasks, states):
        data_name = f'folktables_2018_{task}_{state}'
        data = get_data(f'folktables_datasets/{data_name}-mixed-train',
                        domain_name=f'folktables_datasets/domain/{data_name}-mix')

        # stats_module = TwoWayPrefix.get_stat_module(data.domain, num_rand_queries=1000000)
        stats_module = Marginals.get_all_kway_mixed_combinations(data.domain, 3, bins=[2, 4, 8, 16])

        stats_module.fit(data)
        privga = PrivGA(
            num_generations=20000,
            stop_loss_time_window=100,
            print_progress=False,
            strategy=SimpleGAforSyncData(domain=data.domain,
                                         population_size=1000,
                                         elite_size=10,
                                         data_size=2000,
                                         muta_rate=1,
                                         mate_rate=100))

        run_experiments(data=data,  algorithm=privga, stats_module=stats_module, epsilon=EPSILON, save_dir=('sync_data', data_name, 'PrivGA'))

        data_disc = data.discretize(num_bins=16)
        train_stats_module = Marginals.get_all_kway_combinations(data_disc.domain, 3)
        train_stats_module.fit(data_disc)

        rap = RelaxedProjection(domain=data_disc.domain, data_size=1000, iterations=5000, learning_rate=0.01,
                                print_progress=False)
        run_experiments(data=data_disc,  algorithm=rap, stats_module=stats_module, epsilon=EPSILON, save_dir=('sync_data', data_name, 'PrivGA'))

