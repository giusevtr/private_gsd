import itertools

import jax.random
import matplotlib.pyplot as plt
import pandas as pd
import os
from models import GSD, SimpleGAforSyncData, RelaxedProjectionPP_v3 as RelaxedProjectionPP
from stats import ChainedStatistics, Halfspace, HalfspaceDiff, Prefix, Marginals, PrefixDiff
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

    module_name = 'Halfspaces'
    Res = []
    for data_name in DATA:

        root_path = '../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
        config = load_domain_config(data_name, root_path=root_path)
        df_train = load_df(data_name, root_path=root_path, idxs_path='seed0/train')
        df_test = load_df(data_name, root_path=root_path, idxs_path='seed0/test')
        print(f'train size: {df_train.shape}')
        print(f'test size:  {df_test.shape}')
        domain = Domain.fromdict(config)
        data = Dataset(df_train, domain)
        num_real = len(domain.get_numeric_cols())
        print(f'{data_name} has {num_real} real features.')
        d = domain.get_dimension() - num_real
        # Create statistics and evaluate
        key = jax.random.key(0)

        max_num_queries = 200000
        halfspaces = max_num_queries // d
        module0 = Marginals.get_kway_categorical(data.domain, k=1)
        module1 = Halfspace.get_kway_random_halfspaces(domain=data.domain, k=1, rng=key, random_hs=halfspaces)
        stat_module = ChainedStatistics([module0, module1])
        stat_module.fit(data)
        true_stats = stat_module.get_all_true_statistics()
        stat_fn = stat_module.get_dataset_statistics_fn(jitted=True)

        for seed in [0, 1, 2]:
            for eps in [0.07, 1.00]:
                path = f'../sync_data/rap_results/{data_name}/RP/{eps:.2f}/one_shot/10000/syndata_{seed}.csv'
                df = pd.read_csv(path)
                sync_data = Dataset(df, domain)
                errors = jnp.abs(true_stats - stat_fn(sync_data))
                print(f'RAP(Ranges) ({data_name, module_name}): eps={eps:.2f}, seed={seed}'
                      f'\t max error = {errors.max():.5f}'
                      f'\t avg error = {errors.mean():.5f}')
                Res.append(['RAP', data_name, module_name, 'oneshot', 'oneshot', eps, seed, 'Max', errors.max()])
                Res.append(['RAP', data_name, module_name, 'oneshot', 'oneshot', eps, seed, 'Average', errors.mean()])



    columns = ['Generator', 'Data', 'Statistics', 'T', 'S', 'epsilon', 'seed', 'error type', 'error']
    results = pd.DataFrame(Res, columns=columns)
    os.makedirs('icml_results/', exist_ok=True)
    file_name = 'icml_results/rap_halfspaces.csv'
    results.to_csv(file_name, index=False)

