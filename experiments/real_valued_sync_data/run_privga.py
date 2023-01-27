import itertools
import folktables
import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSEmployment
from utils import Dataset, Domain, DataTransformer
from models import Generator, PrivGAfast, SimpleGAforSyncDataFast, RelaxedProjection
from stats import Marginals
from utils.utils_data import get_data
from experiments.experiment import run_experiments
import os
import jax
EPSILON = (0.07,0.23,0.52,0.74,1.0,)
# EPSILON = (0.01, )
SEEDS = [0]
# adaptive_rounds = (1, 4, 8, 16)
adaptive_rounds = (10,20,30,40,50,60)

if __name__ == "__main__":
    # df = folktables.col_range

    tasks = ['mobility']
    # tasks = ['income']
    # tasks = [ 'mobility', 'travel']
    states = ['CA']
    print("test111:",SEEDS[0])
    for task, state in itertools.product(tasks, states):
        data_name = f'folktables_2018_{task}_{state}'
        data = get_data(f'folktables_datasets/{data_name}-mixed-train',
                        domain_name=f'folktables_datasets/domain/{data_name}-mixed', root_path='../../data_files')

        # data, col_range = data.normalize_real_values()
        # stats_module = TwoWayPrefix.get_stat_module(data.domain, num_rand_queries=1000000)

        num_numeric_feats = len(data.domain.get_numeric_cols())
        stats_module, kway_combinations = Marginals.get_all_kway_mixed_combinations(data.domain, k_disc=1, k_real=1,
                                                                                    bins=[2, 4, 8, 16, 32])

        stats_module.fit(data)
        print(f'Workloads = {len(stats_module.true_stats)}')
        privga = PrivGAfast(
            num_generations=500000,
            print_progress=True,
            strategy=SimpleGAforSyncDataFast(domain=data.domain,
                                         population_size=1000,
                                         elite_size=2,
                                         data_size=2000,
                                         muta_rate=1,
                                         mate_rate=1))

        run_experiments(data=data,  algorithm=privga, stats_module=stats_module, epsilon=EPSILON,
                        adaptive_rounds=adaptive_rounds,
                        seeds=SEEDS,
                        save_dir=('results', data_name, 'PrivGA'),
                        # data_post_processing=lambda data_in: data_in.inverse_map_real_values(col_range)
                        )
