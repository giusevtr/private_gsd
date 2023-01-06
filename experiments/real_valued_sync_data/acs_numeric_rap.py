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
adaptive_rounds = (1, 2, 3, 4)

if __name__ == "__main__":
    # df = folktables.

    # tasks = ['employment', 'coverage', 'income', 'mobility', 'travel']
    tasks = [ 'income']
    states = ['CA']

    for task, state in itertools.product(tasks, states):
        data_name = f'folktables_2018_{task}_{state}'
        data = get_data(f'folktables_datasets/{data_name}-mixed-train',
                        domain_name=f'folktables_datasets/domain/{data_name}-num')

        data, col_range = data.normalize_real_values()
        # stats_module = TwoWayPrefix.get_stat_module(data.domain, num_rand_queries=1000000)

        num_numeric_feats = len(data.domain.get_numeric_cols())
        K = min(num_numeric_feats, 2)
        stats_module, kway_combinations = Marginals.get_all_kway_mixed_combinations(data.domain, k_disc=0, k_real=K,
                                                                                    bins=[2, 4, 8, 16, 32])
        #######
        ## RAP
        #######
        data_disc = data.discretize(num_bins=32)
        # train_stats_module = Marginals.get_all_kway_combinations(data_disc.domain, 3)
        train_stats_module = Marginals(data_disc.domain, kway_combinations=kway_combinations)
        train_stats_module.fit(data_disc)

        numeric_features = data.domain.get_numeric_cols()
        rap_post_processing = lambda data: Dataset.to_numeric(data, numeric_features)


        rap = RelaxedProjection(domain=data_disc.domain, data_size=500, iterations=5000, learning_rate=0.001,
                                print_progress=False)
        run_experiments(data=data_disc,  algorithm=rap, stats_module=train_stats_module, epsilon=EPSILON,
                        adaptive_rounds=adaptive_rounds,
                        save_dir=('real_valued_sync_data', data_name, 'RAP'),
                        data_post_processing=rap_post_processing)

