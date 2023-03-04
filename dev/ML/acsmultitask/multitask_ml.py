import jax.random
import matplotlib.pyplot as plt
import pandas as pd

from models import GeneticSD, GeneticStrategy
from stats import ChainedStatistics, Halfspace, Marginals
# from utils.utils_data import get_data
import jax.numpy as jnp
# from dp_data.data import get_data
from dp_data import load_domain_config, load_df, get_evaluate_ml
from utils import timer, Dataset, Domain
from utils.cdp2adp import cdp_rho, cdp_eps
import numpy as np


def filter_outliers(df_train, df_test):
    domain = Domain.fromdict(config)

    df = df_train.append(df_test)

    for num_col in domain.get_numeric_cols():
        q_lo = df[num_col].quantile(0.01)
        q_hi = df[num_col].quantile(0.95)
        size1 = len(df_train)
        df_train_filtered = df_train[(df_train[num_col] <= q_hi) & (df_train[num_col] >= q_lo)]
        df_test_filtered = df_test[(df_test[num_col] <= q_hi) & (df_test[num_col] >= q_lo)]
        size2 = len(df_train_filtered)
        print(f'Numeric column={num_col}. Removing {size1 - size2} rows.')
        # df_filtered = df[(df[num_col] <= q_hi)]
        df_train = df_train_filtered
        df_test = df_test_filtered

    for num_col in domain.get_numeric_cols():
        maxv = df_train[num_col].max()
        minv = df_train[num_col].min()
        meanv = df_train[num_col].mean()
        print(f'Col={num_col:<10}: mean={meanv:<5.3f}, min={minv:<5.3f},  max={maxv:<5.3f},')
        # df_train[num_col].hist()
        # plt.yscale('log')
        # plt.show()

    return df_train, df_test



if __name__ == "__main__":
    dataset_name = 'folktables_2018_multitask_NY'
    root_path = '../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)
    config.pop('JWMNP_bin')
    config.pop('ESR')
    config.pop('MIG')
    df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
    df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')

    # df_train, df_test = filter_outliers(df_train, df_test)

    # df_train = df_train.sample(n=20000)
    # df_test = df_test.sample(n=1000)

    print(f'train size: {df_train.shape}')
    print(f'test size:  {df_test.shape}')
    # domain = Domain.fromdict(config)
    # data = Dataset(df_train, domain)
    # targets = ['JWMNP_bin', 'PINCP', 'ESR', 'MIG', 'PUBCOV']
    targets = ['PINCP', 'PUBCOV']


    ml_eval_fn = get_evaluate_ml(df_test, config, targets=targets, models=['LogisticRegression'])

    results = ml_eval_fn(df_train, 0)
    print(f'Original train:')
    # results = results[results['Metric'] == 'f1_macro']
    print(results)

    # df_sync1 = pd.read_csv('folktables_2018_multitask_NY_sync_0.07_0.csv')
    # df_sync2 = pd.read_csv('folktables_2018_multitask_NY_sync_1.00_0.csv')
    #
    # print('Synthetic train eps=0.07:')
    # results = ml_eval_fn(df_sync1, 0)
    # results = results[results['Metric'] == 'f1']
    # print(results)
    #
    # print('Synthetic train eps=1.00:')
    # results = ml_eval_fn(df_sync2, 0)
    # results = results[results['Metric'] == 'f1']
    # print(results)
