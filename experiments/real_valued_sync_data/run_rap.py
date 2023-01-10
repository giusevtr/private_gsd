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
# EPSILON = (0.01, )
SEEDS = [0, 1, 2]
adaptive_rounds = (1, 2, 3, 4, 5)

if __name__ == "__main__":
    # df = folktables.

    tasks = ['employment', 'coverage', 'income', 'mobility', 'travel']
    # tasks = ['income']
    states = ['CA']

    for task, state in itertools.product(tasks, states):
        data_name = f'folktables_2018_{task}_{state}'
        data = get_data(f'folktables_datasets/{data_name}-mixed-train',
                        domain_name=f'folktables_datasets/domain/{data_name}-num')

        # Normalize numeric columns to be in [0, 1]
        data, col_range = data.normalize_real_values()

        #  Get range-marginals queries.
        num_numeric_feats = len(data.domain.get_numeric_cols())
        stats_module, kway_combinations = Marginals.get_all_kway_mixed_combinations(data.domain, k_disc=0, k_real=2,
                                                                                    bins=[2, 4, 8, 16, 32])

        # Discretize data using binning strategy
        data_disc = data.discretize(num_bins=32)

        # Get categorical-marginals queries for the discretized data
        train_stats_module = Marginals(data_disc.domain, kway_combinations=kway_combinations)
        train_stats_module.fit(data_disc)

        # This step is for converting the discrete synthetic data back to numerica domain
        numeric_features = data.domain.get_numeric_cols()
        rap_post_processing = lambda data: Dataset.to_numeric(data, numeric_features)


        # Initialize RAP with discrete domain
        rap = RelaxedProjection(domain=data_disc.domain, data_size=1000, iterations=50000, learning_rate=0.0009,
                                print_progress=True)

        # Run rap with discretized data and categorical-marginal queries
        run_experiments(data=data_disc,  algorithm=rap, stats_module=train_stats_module,
                        epsilon=EPSILON,
                        adaptive_rounds=adaptive_rounds,
                        seeds=SEEDS,
                        save_dir=('results', data_name, 'RAP'),
                        data_post_processing=rap_post_processing)

