import itertools

import jax.random
import matplotlib.pyplot as plt
import pandas as pd
import os
from models import GSD, SimpleGAforSyncData, RelaxedProjectionPP_v3 as RelaxedProjectionPP, RelaxedProjection
from stats import ChainedStatistics, Halfspace, HalfspaceDiff, Prefix, MarginalsDiff, PrefixDiff
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


def run(dataset_name, module_name, seeds=(0, 1, 2), eps_values=(0.07, 0.23, 0.52, 0.74, 1.0)):

    max_num_queries = 200000
    rounds = 50
    num_sample = 5
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
    key = jax.random.PRNGKey(0)
    # module0 = MarginalsDiff.get_all_kway_categorical_combinations(data.domain, k=2)
    num_real = len(domain.get_numeric_cols())
    print(f'{dataset_name} has {num_real} real features.')
    d = domain.get_dimension() - num_real
    module0 = MarginalsDiff.get_all_kway_categorical_combinations(data.domain, k=2)

    if module_name == 'Prefix':
        domain.get_dimension()
        prefixes = max_num_queries // d
        # module1 = PrefixDiff.get_kway_prefixes(domain, k_cat=1, k_num=2, rng=key, random_prefixes=prefixes)
        module1 = PrefixDiff(domain, k_cat=1, k_prefix=2,
                             cat_kway_combinations=[('PINCP',)], rng=key,
                             num_random_prefixes=100000)
        # module2 = PrefixDiff.get_kway_prefixes(domain, k_cat=0, k_num=2, rng=key, random_prefixes=max_num_queries)
    else:
        halfspaces = max_num_queries // d
        module1 = HalfspaceDiff.get_kway_random_halfspaces(domain=data.domain, k=1, rng=key, random_hs=halfspaces)
        # module2 = HalfspaceDiff.get_kway_random_halfspaces(domain=data.domain, k=0, rng=key, random_hs=max_num_queries)
    stat_module = ChainedStatistics([
        module0,
                                     module1])
    stat_module.fit(data)
    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module.get_dataset_statistics_fn()

    algo = RelaxedProjectionPP(domain=data.domain, data_size=1000,
                               iterations=1000,  print_progress=True)

    delta = 1.0 / len(data) ** 2
    for seed in seeds:
        for eps in eps_values:
            key = jax.random.PRNGKey(seed)
            t0 = timer()
            sync_dir = f'sync_data/{dataset_name}/RAP++_old/{module_name}/{rounds}/{num_sample}/{eps:.2f}/'
            os.makedirs(sync_dir, exist_ok=True)
            sync_data = algo.fit_dp_hybrid(key, stat_module=stat_module,
                                           rounds=rounds,
                                           epsilon=eps, delta=delta,
                                           num_sample=num_sample,
                                           )
            sync_data.df.to_csv(f'{sync_dir}/sync_data_{seed}.csv', index=False)
            errors = jnp.abs(true_stats - stat_fn(sync_data))
            print(f'RAP++_old({dataset_name, module_name}): eps={eps:.2f}, seed={seed}'
                  f'\t max error = {errors.max():.5f}'
                  f'\t avg error = {errors.mean():.5f}'
                  f'\t time = {timer() - t0:.4f}')
            Res.append(['RAP++_old', dataset_name, module_name, rounds, num_sample, eps, seed, 'Max', errors.max()])
            Res.append(['RAP++_old', dataset_name, module_name, rounds, num_sample, eps, seed, 'Average', errors.mean()])

        print()

    columns = ['Generator', 'Data', 'Statistics', 'T', 'S', 'epsilon', 'seed', 'error type', 'error']
    results_df = pd.DataFrame(Res, columns=columns)
    return results_df

if __name__ == "__main__":

    DATA = [
        # 'folktables_2018_real_CA',
        # 'folktables_2014_coverage_NY',
        # 'folktables_2018_employment_CA',
        'folktables_2018_income_CA',
        # 'folktables_2018_mobility_CA',
        # 'folktables_2018_travel_CA',
    ]

    MODULE = [
        'Prefix',
        # 'Halfspaces'
    ]

    os.makedirs('icml_results/', exist_ok=True)
    file_name = 'icml_results/rap++.csv'
    results = None
    if os.path.exists(file_name):
        print(f'reading {file_name}')
        results = pd.read_csv(file_name)
    for data, module in itertools.product(DATA, MODULE):
        results_temp = run(data, module, eps_values=[1.00, 0.07])
        results = pd.concat([results, results_temp], ignore_index=True) if results is not None else results_temp
        print(f'Saving {file_name}')
        results.to_csv(file_name, index=False)

