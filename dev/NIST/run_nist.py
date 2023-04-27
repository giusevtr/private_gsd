import os, sys
import jax.random
import pandas as pd
import numpy as np
from models import GeneticSDConsistent as GeneticSD
from stats import ChainedStatistics,  Marginals, NullCounts
import jax.numpy as jnp
from dp_data import load_domain_config, load_df
from dp_data.data_preprocessor import DataPreprocessor
from utils import timer, Dataset, Domain
import pickle
from dev.NIST.consistency import get_consistency_fn
from dev.NIST.consistency_simple import get_nist_simple_consistency_fn
import itertools


NULL_COLS = [
    "MSP",
    "NOC",
    "NPF",
    "DENSITY",
    "INDP",
    "INDP_CAT",
    "EDU",
    "PINCP",
    "PINCP_DECILE",
    "POVPIP",
    "DVET",
    "DREM",
    "DPHY",
    "PWGTP",
    "WGTP"]

inc_bins_pre = jnp.array([-10000, -100, -10, -5,
                         1,
                         5,
                         10, 50, 100, 200,
                         500, 700, 1000, 2000, 3000, 4500, 8000, 9000, 10000,
                         10800, 12000, 14390, 15000, 18010,
                         20000, 23000, 25000, 27800,
                         30000, 33000, 35000, 37000,
                         40000, 45000, 47040,
                         50000, 55020,
                         60000, 65000, 67000,
                         70000, 75000,
                         80000, 85000,
                         90000, 95000,
                         100000, 101300, 140020,
                         200000, 300000, 400000, 500000, 1166000])


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    nist_type = str(sys.argv[2])

    assert nist_type in ['all', 'simple']

    root_path = '../../dp-data-dev/datasets/preprocessed/sdnist_dce/'
    config = load_domain_config(dataset_name, root_path=root_path)
    df_train = load_df(dataset_name, root_path=root_path)

    preprocessor_path = os.path.join(root_path + dataset_name, 'preprocessor.pkl')
    bins = {}
    with open(preprocessor_path, 'rb') as handle:
        # preprocessor:
        preprocessor = pickle.load(handle)
        temp: pd.DataFrame
        preprocessor: DataPreprocessor
        min_val, max_val = preprocessor.mappings_num['PINCP']
        print(min_val, max_val)
        inc_bins = (inc_bins_pre - min_val) / (max_val - min_val)
        bins['PINCP'] = inc_bins

    domain = Domain(config, NULL_COLS, bin_edges=bins)
    data = Dataset(df_train, domain)


    N = len(data.df)
    dataset_name = f'{dataset_name}_{nist_type}'

    eps = float(sys.argv[3])
    data_size_str = sys.argv[4]
    data_size = N if data_size_str == 'N' else int(data_size_str)
    k = int(sys.argv[5])

    print(f'Input data {dataset_name}, epsilon={eps:.2f}, data_size={data_size}, k={k} ')



    all_cols = domain.attrs
    consistency_fn = None
    if nist_type == 'simple':
        all_cols.remove('INDP')
        all_cols.remove('WGTP')
        all_cols.remove('PWGTP')
        all_cols.remove('DENSITY')
        consistency_fn = get_nist_simple_consistency_fn(domain, preprocessor)
    elif nist_type == 'all':
        all_cols.remove('INDP_CAT')
        consistency_fn = get_consistency_fn(domain, preprocessor)

    data = data.project(all_cols)
    domain = data.domain
    # Create statistics and evaluate
    key = jax.random.PRNGKey(0)
    # One-shot queries
    modules = []
    bins = None
    modules.append(Marginals.get_all_kway_combinations(data.domain, k=1, bin_edges=bins, levels=20))
    modules.append(Marginals.get_all_kway_combinations(data.domain, k=2, levels=10, include_feature='PUMA'))
    # modules.append(Marginals.get_all_kway_combinations(data.domain, k=1, levels=10))
    if k == 3:
        modules.append(Marginals.get_all_kway_combinations(data.domain, k=2, levels=5, bin_edges=bins))
        modules.append(Marginals.get_all_kway_combinations(data.domain, k=3, levels=5, bin_edges=bins,
                                                           include_feature='PUMA'))
    elif k == 2:
        modules.append(Marginals.get_all_kway_combinations(data.domain, k=2, levels=5, bin_edges=bins))

    # module_nulls = NullCounts(data.domain, null_cols=NULL_COLS)
    stat_module = ChainedStatistics(modules)
    stat_module.fit(data, max_queries_per_workload=2000)

    true_stats = stat_module.get_all_true_statistics()
    stat_fn0 = stat_module._get_workload_fn()

    N = len(data.df)
    algo = GeneticSD(num_generations=1000000,
                       print_progress=True,
                       stop_early=True,
                       domain=data.domain,
                       population_size=100,
                       data_size=data_size,
                        stop_early_gen=data_size,
                       inconsistency_fn=consistency_fn,
                       mate_perturbation=1e-4,
                       null_value_frac=0.01,
                       )
    # Choose algorithm parameters

    # delta = 1.0 / len(data) ** 2
    delta = 10**(-5)
    # Generate differentially private synthetic data with ADAPTIVE mechanism
    for seed in [0]:
        key = jax.random.PRNGKey(seed)
        t0 = timer()

        sync_data = algo.fit_dp(key, stat_module=stat_module,
                           epsilon=eps,
                           delta=delta)

        sync_dir = f'sync_data/{dataset_name}/{k}/{eps:.2f}/{data_size_str}/oneshot'
        os.makedirs(sync_dir, exist_ok=True)
        print(f'Saving {sync_dir}/sync_data_{seed}.csv')
        sync_data.df.to_csv(f'{sync_dir}/sync_data_{seed}.csv', index=False)
        errors = jnp.abs(true_stats - stat_fn0(sync_data.to_numpy()))

        print(f'Input data {dataset_name}, k={k}, epsilon={eps:.2f}, data_size={data_size}, seed={seed}')
        print(f'GSD(oneshot):'
              f'\t max error = {errors.max():.5f}'
              f'\t avg error = {errors.mean():.6f}'
              f'\t time = {timer() - t0:.4f}')

    print()