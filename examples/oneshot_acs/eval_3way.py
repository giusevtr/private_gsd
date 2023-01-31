import os

import jax.random
import pandas as pd

from models import PrivGA, SimpleGAforSyncData
from stats import Marginals
from utils.utils_data import get_data
from utils import timer
from toy_datasets.classification import get_classification
import jax.numpy as jnp
from utils.plot_low_dim_data import plot_2d_data

from utils import Dataset

if __name__ == "__main__":

    tasks = ['mobility', 'coverage', 'income', 'employment', 'travel']
    # Get Data
    # data_name = f'folktables_2018_CA'
    data_name = 'folktables_2018_mobility_CA'
    data = get_data(f'{data_name}-mixed-train', domain_name=f'domain/{data_name}-cat',  root_path='../../data_files/folktables_datasets')
    # data = get_classification(DATA_SIZE=1000)

    SYNC_DATA_SIZE = 2000
    # Create statistics and evaluate
    marginal_module, _ = Marginals.get_all_kway_combinations(data.domain, k=3, bins=[2, 4, 8, 16, 32])
    marginal_module.fit(data)

    true_stats = marginal_module.get_true_statistics()
    stat_fn = marginal_module._get_workload_fn()

    # rap = RelaxedProjection(domain=data.domain, data_size=1000, iterations=1000, learning_rate=0.05,
    #                         print_progress=False)
    delta = 1.0 / len(data) ** 2
    # Generate differentially private synthetic data with ADAPTIVE mechanism
    Res = []
    for eps in [0.07, 0.23, 0.52, 0.74, 1.0]:
        for seed in [0, 1, 2]:

            path_dir = f'sync_data/{data_name}/{eps}/'
            sync_path = f'{path_dir}/sync_data_{seed}.csv'
            df = pd.read_csv(sync_path, index_col=None)

            sync_data = Dataset(df, data.domain)

            errors = jnp.abs(true_stats - stat_fn(sync_data.to_numpy()))
            print(f'PrivGA: eps={eps:.2f}, seed={seed}'
                  f'\t max error = {errors.max():.5f}'
                  f'\t avg error = {errors.mean():.5f}'
                  )
            Res.append(['PrivGA', eps, seed, float(errors.max()), float(errors.mean())])

        print()
    cols = ['generator', 'epsilon', 'seed', 'error_max', 'error_mean']
    res_df = pd.DataFrame(Res, columns=cols)
    res_df.to_csv('privga_3way_results.csv', index=False)