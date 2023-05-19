import itertools

import jax.random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from models import PrivGA, SimpleGAforSyncData, PrivGAJit
from stats import ChainedStatistics, Marginals
# from utils.utils_data import get_data
from utils import timer
import jax.numpy as jnp
# from dp_data.data import get_data
from utils import timer, Dataset, Domain , get_Xy, filter_outliers
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression

from dp_data import load_domain_config, load_df

from dev.dataloading.data_functions.acs import get_acs_all


def get_cat_marginals(data: Dataset, k):
    domain = data.domain
    kway_combinations = [list(idx) for idx in itertools.combinations(domain.get_categorical_cols(), k)]
    A = []
    for cols in kway_combinations:
        answers = data.project(cols).datavector()
        A.append(answers)
    return np.concatenate(A) / len(data.df)


def run(dataset_name,  seeds=(0, 1, 2), eps_values=(0.07, 0.23, 0.52, 0.74, 1.0),
        T=(100, ),
        eval_only=False):
    module_name = '3-Cat'
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

    true_stats = get_cat_marginals(data, k=3)

    print(f'{dataset_name} has {len(domain.get_numeric_cols())} real features and '
          f'{len(domain.get_categorical_cols())} cat features.')
    print(f'Data cardinality is {domain.size()}.')
    print(f'Number of queries is {true_stats.shape[0]}.')

    delta = 1.0 / len(data) ** 2
    for seed in seeds:
        for eps in eps_values:
            for rounds in T:
                t0 = timer()
                sync_dir = f'sync_data/{dataset_name}/GSD/{module_name}/{rounds}/1/{eps:.2f}/'
                os.makedirs(sync_dir, exist_ok=True)
                sync_path = f'{sync_dir}/sync_data_{seed}.csv'
                if not os.path.exists(sync_path): continue
                sync_df = pd.read_csv(sync_path)
                sync_data = Dataset(sync_df, domain)

                sync_stats = get_cat_marginals(sync_data, k=3)

                errors = jnp.abs(true_stats - sync_stats)
                elapsed_time = timer() - t0
                print(f'GSD({dataset_name, module_name}): eps={eps:.2f}, seed={seed}'
                      f'\t T = {rounds}'
                      f'\t max error = {errors.max():.5f}'
                      f'\t avg error = {errors.mean():.5f}'
                      f'\t time = {elapsed_time:.4f}')
                Res.append(['GSD', dataset_name, module_name, rounds, 1, eps, seed, 'Max', errors.max(), elapsed_time])
                Res.append(['GSD', dataset_name, module_name, rounds, 1, eps, seed, 'Average', errors.mean(), elapsed_time])

        print()

    columns = ['Generator', 'Data', 'Statistics', 'T', 'S', 'epsilon', 'seed', 'error type', 'error', 'time']
    results_df = pd.DataFrame(Res, columns=columns)
    return results_df

if __name__ == "__main__":

    DATA = [
        'folktables_2018_travel_CA',
        'folktables_2018_income_CA',
        'folktables_2018_coverage_CA',
        'folktables_2018_mobility_CA',
        'folktables_2018_employment_CA',
    ]

    T = [25, 50, 75, 100]
    os.makedirs('icml_results/', exist_ok=True)
    results = None
    for data in DATA:
        file_name = f'icml_results/gsd_adaptive_3way_categorical_{data}.csv'
        results_temp = run(data, eps_values=[1, 0.74, 0.52, 0.23, 0.07], seeds=[0, 1, 2],
                           T=T)
        results = pd.concat([results, results_temp], ignore_index=True) if results is not None else results_temp
        print(f'Saving: {file_name}')
        results.to_csv(file_name, index=False)

