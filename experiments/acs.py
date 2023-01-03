import itertools
import folktables
import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSEmployment
from utils import Dataset, Domain, DataTransformer
from models import Generator, PrivGA, SimpleGAforSyncData, RelaxedProjection
from stats import Marginals
from utils.utils_data import get_data
from experiments.experiment import run_experiments
import os
import jax
EPSILON = (0.07, 0.23, 0.52, 0.74, 1.0)

if __name__ == "__main__":
    # df = folktables.

    # tasks = ['employment', 'coverage', 'income', 'mobility', 'travel']
    tasks = [ 'mobility', 'travel']
    states = ['CA']

    for task, state in itertools.product(tasks, states):
        data_name = f'folktables_2018_{task}_{state}'
        data = get_data(f'folktables_datasets/{data_name}-mixed-train',
                        domain_name=f'folktables_datasets/domain/{data_name}-mixed')

        # stats_module = TwoWayPrefix.get_stat_module(data.domain, num_rand_queries=1000000)
        stats_module = Marginals.get_all_kway_mixed_combinations(data.domain, 3, bins=[2, 4, 8, 16, 32])

        stats_module.fit(data)
        privga = PrivGA(
            num_generations=20000,
            stop_loss_time_window=100,
            print_progress=False,
            strategy=SimpleGAforSyncData(domain=data.domain,
                                         population_size=50,
                                         elite_size=2,
                                         data_size=200,
                                         muta_rate=1,
                                         mate_rate=10))

        run_experiments(data=data,  algorithm=privga, stats_module=stats_module, epsilon=EPSILON,
                        save_dir=('sync_path', data_name, 'PrivGA'))

        #######
        ## RAP
        #######
        data_disc = data.discretize(num_bins=32)
        train_stats_module = Marginals.get_all_kway_combinations(data_disc.domain, 3)
        train_stats_module.fit(data_disc)

        numeric_features = data.domain.get_numeric_cols()
        rap_post_processing = lambda data: Dataset.to_numeric(data, numeric_features)


        rap = RelaxedProjection(domain=data_disc.domain, data_size=500, iterations=5000, learning_rate=0.01,
                                print_progress=False)
        run_experiments(data=data_disc,  algorithm=rap, stats_module=stats_module, epsilon=EPSILON,
                        save_dir=('sync_path', data_name, 'RAP'),
                        data_post_processing=rap_post_processing)

