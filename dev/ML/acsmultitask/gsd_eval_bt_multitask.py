import itertools
import os.path
import pickle

import jax.random
import matplotlib.pyplot as plt
import pandas as pd

from models import GSD, SimpleGAforSyncData
from stats import ChainedStatistics, Halfspace, Marginals
# from utils.utils_data import get_data
import jax.numpy as jnp
# from dp_data.data import get_data
from dp_data import load_domain_config, load_df, DataPreprocessor, ml_eval
from utils import timer, Dataset, Domain
from utils.cdp2adp import cdp_rho, cdp_eps
import numpy as np
from dev.ML.ml_utils import evaluate_machine_learning_task

from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    dataset_name = 'folktables_2018_multitask_CA'
    root_path = '../../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)
    df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
    df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')

    preprocesor: DataPreprocessor
    preprocesor = pickle.load(open(f'{root_path}/{dataset_name}/preprocessor.pkl', 'rb'))

    domain = Domain.fromdict(config)
    data = Dataset(df_train, domain)
    cat_cols = domain.get_categorical_cols()
    num_cols = domain.get_numeric_cols()

    print(f'train size: {df_train.shape}')
    print(f'test size:  {df_test.shape}')

    epsilon_vals = [0.07, 0.23, 0.52, 0.74, 1]
    seeds = [0, 1, 2]

    # Evaluate BT Queries
    module = Marginals.get_all_kway_combinations(domain, k=2, bins=[2, 4, 8, 16, 32])
    stat_fn = module._get_dataset_statistics_fn(jitted=True)
    true_bt_stats = stat_fn(data)

    queries = [
        # ('Binary_Tree_Marginals', 'BT'),
        # ('Histogram', 'Hist')
        ('2Cat+HS_0.8', 'HS')
    ]
    Res = []
    for eps, seed, (q_name, q_short_name) in itertools.product(epsilon_vals, seeds, queries):

        sync_path = f'sync_data/GSD/{q_name}/50/5/{eps:.2f}/sync_data_{seed}.csv'
        if not os.path.exists(sync_path):
            print(f'{sync_path} NOT FOUND')
            continue

        print(f'reading {sync_path}')
        df_sync_post = pd.read_csv(sync_path)

        sync_data = Dataset(df_sync_post, domain)
        sync_stats = stat_fn(sync_data)

        errors = np.abs(sync_stats - true_bt_stats)

        max_error = errors.max()
        avg_error = errors.mean()

        print(f'eps={eps:.2f}, max={ errors.max():.3f}, avg={errors.mean():.7f}')
        Res.append(['GSD', dataset_name, 'BT', 50, 5, eps, seed, 'Max', errors.max(), 0])
        Res.append(['GSD', dataset_name, 'BT', 50, 5, eps, seed, 'Average', errors.mean(), 0])

    columns = ['Generator', 'Data', 'Statistics', 'T', 'S', 'epsilon', 'seed', 'error type', 'error', 'time']
    results_df = pd.DataFrame(Res, columns=columns)
    print(results_df)
    file_path = 'results'
    os.makedirs(file_path, exist_ok=True)