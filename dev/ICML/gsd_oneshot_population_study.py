import itertools

import jax.random
import matplotlib.pyplot as plt
import pandas as pd
import os
from models import GSD
from stats import ChainedStatistics, Marginals
# from utils.utils_data import get_data
from utils import timer
import jax.numpy as jnp
# from dp_data.data import get_data
from utils import timer, Dataset, Domain
from dp_data import load_domain_config, load_df


def run(dataset_name, module_name):
    Res = []

    root_path = '../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)
    df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
    df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')

    print(f'train size: {df_train.shape}')
    print(f'test size:  {df_test.shape}')
    domain = Domain.fromdict(config)
    data = Dataset(df_train, domain)

    # Create statistics and evaluate
    # module0 = MarginalsDiff.get_all_kway_categorical_combinations(data.domain, k=2)

    module = Marginals.get_all_kway_combinations(domain, k=2, bins=[2, 4, 8, 16, 32])
    stat_module = ChainedStatistics([module])
    stat_module.fit(data)
    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module.get_dataset_statistics_fn()

    print(f'{dataset_name} has {len(domain.get_numeric_cols())} real features and '
          f'{len(domain.get_categorical_cols())} cat features.')
    print(f'Data cardinality is {domain.size()}.')
    print(f'Number of queries is {true_stats.shape[0]}.')

    delta = 1.0 / len(data) ** 2
    for pop_size in [5, 10, 20, 40, 80, 160, 320, 640, 1280]:
        # algo = PrivGA(num_generations=500000,
        #               strategy=SimpleGAforSyncData(domain, 2000, population_size=pop_size, muta_rate=1, mate_rate=1),
        #               print_progress=False,
        #               stop_eary_threshold=0.014)
        algo = GSD(num_generations=2000000, stop_early=True,
                   domain=domain, data_size=2000, population_size=pop_size, muta_rate=1, mate_rate=1,
                   print_progress=False,
                   # stop_eary_threshold=0.1,
                   # stop_eary_threshold = 0.014
                   sparse_statistics=True
                   )

        key = jax.random.key(0)
        t0 = timer()
        sync_data = algo.fit_dp(key, stat_module=stat_module,
                                       epsilon=1, delta=delta,

                                       )
        errors = jnp.abs(true_stats - stat_fn(sync_data))
        elapsed_time = timer() - t0
        print(f'{algo}, GSD({dataset_name, module_name}): '
              f'data_size={20000}, '
              f'pop_size={pop_size}, '
              f'eps={1:.2f}, seed={0}'
              f'\t max error = {errors.max():.5f}'
              f'\t avg error = {errors.mean():.6f}'
              f'\t time = {elapsed_time:.4f}')

        print()

    columns = ['Generator', 'Data', 'Statistics', 'T', 'S', 'epsilon', 'seed', 'error type', 'error', 'time']
    results_df = pd.DataFrame(Res, columns=columns)
    return results_df

if __name__ == "__main__":

    DATA = [
        # 'folktables_2018_real_CA',
        'folktables_2018_coverage_CA',
        'folktables_2018_employment_CA',
        'folktables_2018_income_CA',
        'folktables_2018_mobility_CA',
        'folktables_2018_travel_CA',
    ]

    # os.makedirs('icml_results/', exist_ok=True)
    # file_name = 'icml_results/oneshot_ranges/gsd_oneshot.csv'
    # results = None
    # if os.path.exists(file_name):
    #     print(f'reading {file_name}')
    #     results = pd.read_csv(file_name)
    for data in DATA:
        results_temp = run(data, 'Ranges')

