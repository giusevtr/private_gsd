import itertools

import jax.random
import pandas as pd
import os
from stats import ChainedStatistics, Halfspace,  Prefix, Marginals
import jax.numpy as jnp
from utils import timer, Dataset, Domain , get_Xy, filter_outliers
from dp_data import load_domain_config, load_df


def run(dataset_name, module_name, seeds=(0, 1, 2), num_samples=(10, 100, 1000)):

    max_num_queries = 200000
    rounds = 10
    num_sample = 1000
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
    key = jax.random.key(0)
    # module0 = MarginalsDiff.get_all_kway_categorical_combinations(data.domain, k=2)
    num_real = len(domain.get_numeric_cols())
    num_cat = len(domain.get_categorical_cols())
    rows = data.df.shape[0]
    print(f'{dataset_name} has {num_real} real features and {num_cat} and rows={rows}')
    d = domain.get_dimension() - num_real

    module0 = Marginals.get_kway_categorical(data.domain, k=1)

    if module_name == 'Prefix':
        domain.get_dimension()
        prefixes = max_num_queries // d
        # module1 = Prefix.get_kway_prefixes(domain, k_cat=0, k_num=1, rng=key, random_prefixes=prefixes)
        module2 = Prefix.get_kway_prefixes(domain, k_cat=1, k_num=2, rng=key, random_prefixes=prefixes)
    else:
        halfspaces = max_num_queries // d
        module2 = Halfspace.get_kway_random_halfspaces(domain=data.domain, k=1, rng=key, random_hs=halfspaces)
    stat_module = ChainedStatistics([module0, module2])
    stat_module.fit(data)
    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module.get_dataset_statistics_fn()


    delta = 1.0 / len(data) ** 2
    for seed in seeds:
        for samples in num_samples:
            key = jax.random.key(seed)
            t0 = timer()

            sync_data = data.sample(n=samples, seed=seed)
            errors = jnp.abs(true_stats - stat_fn(sync_data))
            print(f'Sample({dataset_name, module_name}): samples={samples:.2f}, seed={seed}'
                  f'\t max error = {errors.max():.5f}'
                  f'\t avg error = {errors.mean():.5f}'
                  f'\t time = {timer() - t0:.4f}')
            Res.append(['Sampling', dataset_name, module_name, samples, seed, 'Max', errors.max()])
            Res.append(['Sampling', dataset_name, module_name, samples, seed, 'Average', errors.mean()])

        print()

    columns = ['Generator', 'Data', 'Statistics', 'samples', 'seed', 'error type', 'error']
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

    MODULE = [
        'Prefix',
        'Halfspaces'
    ]

    os.makedirs('icml_results/', exist_ok=True)
    file_name = 'icml_results/sampling.csv'
    results = None
    if os.path.exists(file_name):
        print(f'reading {file_name}')
        results = pd.read_csv(file_name)
    for data, module in itertools.product(DATA, MODULE):
        results_temp = run(data, module, num_samples=[100, 1000, 10000])
        results = pd.concat([results, results_temp], ignore_index=True) if results is not None else results_temp
        print(f'Saving {file_name}')
        results.to_csv(file_name, index=False)

