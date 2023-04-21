import os, sys
import jax.random
import pandas as pd
import numpy as np
from models import GeneticSDConsistent as GeneticSD
from stats import ChainedStatistics,  Marginals, NullCounts, MarginalsV2, MarginalsV3
import jax.numpy as jnp
from dp_data import load_domain_config, load_df
from dp_data.data_preprocessor import DataPreprocessor
from utils import timer, Dataset, Domain
import pickle
from dev.NIST.consistency import get_consistency_fn
from dev.NIST.consistency_simple import get_nist_simple_consistency_fn
import itertools
from tqdm import  tqdm

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

inc_bins_pre = np.array([-10000, -100, -10, -5,
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
    eps = int(sys.argv[3])
    data_size = int(sys.argv[4])

    assert nist_type in ['all', 'simple']

    print(f'Input data {dataset_name}, epsilon={eps:.2f}, data_size={data_size} ')
    root_path = '../../dp-data-dev/datasets/preprocessed/sdnist_dce/'
    config = load_domain_config(dataset_name, root_path=root_path)
    df_train = load_df(dataset_name, root_path=root_path)

    domain = Domain(config, NULL_COLS)
    data = Dataset(df_train, domain)
    preprocessor_path = os.path.join(root_path + dataset_name, 'preprocessor.pkl')
    dataset_name = f'{dataset_name}_{nist_type}'


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
    # module0 = Marginals.get_all_kway_combinations(data.domain, k=1, bins=bins, levels=5)
    # kway_combinations = [list(idx) for idx in itertools.combinations(domain.attrs, 3)
    #                      if 'PUMA' in list(idx)]
    # module1 = Marginals(data.domain, k=3, kway_combinations=kway_combinations, levels=5)

    # module0 = MarginalsV2.get_kway_categorical(data.domain, k=1)

    t0 = timer()
    module0 = Marginals.get_kway_categorical(data.domain, k=4)
    stat_fn0 = jax.jit(module0._get_workload_fn())
    X = data.to_numpy()
    X_temp = X[:100, :]
    for _ in tqdm(range(10000), desc='running stat_fn0'):
        stat_0 = stat_fn0(X_temp).block_until_ready()
    print(f'time0=', timer() - t0, 'stat_0.shape', stat_0.shape)

    t0 = timer()
    module1 = MarginalsV2.get_kway_categorical(data.domain, k=4)
    # stat_fn1 = module1._get_dataset_statistics_fn()
    stat_fn1 = module1._get_workload_fn()
    X = data.to_numpy_np()

    X_temp = X[:100, :]
    for _ in tqdm(range(10000), desc='running stat_fn1'):

        stat_1 = stat_fn1(X_temp)
        # stat_1 = jnp.stack([stat_fn1(X_temp[i]) for i in range(100)])
    print(f'time1=', timer() - t0, 'stat_1.shape', stat_1.shape)

