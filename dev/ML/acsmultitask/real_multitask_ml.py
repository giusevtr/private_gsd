import itertools
import os.path
import pickle

import jax.random
import matplotlib.pyplot as plt
import pandas as pd

from models import PrivGA, SimpleGAforSyncData
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
    cat_cols = domain.get_categorical_cols()
    num_cols = domain.get_numeric_cols()

    print(f'train size: {df_train.shape}')
    print(f'test size:  {df_test.shape}')
    # domain = Domain.fromdict(config)
    # data = Dataset(df_train, domain)
    targets = ['JWMNP_bin', 'PINCP', 'ESR', 'MIG', 'PUBCOV']
    # targets = ['PINCP', 'PUBCOV']
    features = []
    for f in domain.attrs:
        if f not in targets:
            features.append(f)
    ml_fn = ml_eval.get_evaluate_ml(df_test, config, targets, models=['LogisticRegression'])

    epsilon_vals = [0.07, 0.23, 0.52, 0.74, 1]
    seeds = [0, 1, 2]

    Res = []
    res = ml_fn(df_train, seed=0)
    res = res[res['Eval Data'] == 'Test']
    res = res[res['Metric'] == 'f1_macro']
    print(res)
    for i, row in res.iterrows():
        target = row['target']
        f1 = row['Score']
        for eps in epsilon_vals:
            Res.append([dataset_name, 'No', f'Original', '', 'oneshot', 'oneshot', 'LR', target, eps, 'F1', 0, f1])
        # Res.append([dataset_name, 'Yes', algo_name+query_name, 'LR', target, eps, 'Accuracy', seed, acc])

    results = pd.DataFrame(Res, columns=['Data', 'Is DP', 'Generator',
                                         'Statistics',
                                         'T', 'S',
                                         'Model',
                                         'Target', 'epsilon', 'Metric',
                                         'seed',
                                         'Score'])

    print(results)
    file_path = 'results'
    os.makedirs(file_path, exist_ok=True)
    file_path = f'results/results_original.csv'
    # if os.path.exists(file_path):
    #     results_pre = pd.read_csv(file_path, index_col=None)
    #     results = results_pre.append(results)
    print(f'Saving ', file_path)
    results.to_csv(file_path, index=False)