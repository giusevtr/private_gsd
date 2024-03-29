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
    targets = ['JWMNP_bin', 'PINCP', 'MIG', 'PUBCOV', 'ESR']
    # targets = ['PINCP', 'PUBCOV']
    features = []
    for f in domain.attrs:
        if f not in targets:
            features.append(f)

    model = 'LogisticRegression'
    # model = 'XGBoost'
    # model = 'RandomForest'
    ml_fn = ml_eval.get_evaluate_ml(df_test, config, targets, models=[model])

    epsilon_vals = [0.07, 0.23, 0.52, 0.74, 1]
    seeds = [0, 1, 2]


    queries = [
        ('Binary_Tree_Marginals', 'BT'),
        # ('Histogram', 'Hist')
        # ('2Cat+HS_0.8_10m', 'HS')
        # ('2Cat+Prefix', 'Prefix')
    ]
    Res = []
    for eps, seed, (q_name, q_short_name) in itertools.product(epsilon_vals, seeds, queries):

        # sync_path = f'sync_data/GSD/folktables_2018_multitask_CA/GSD/{q_name}/50/5/{eps:.2f}/sync_data_{seed}.csv'
        sync_path = f'sync_data/GSD/folktables_2018_multitask_CA/GSD/{q_name}/{eps:.2f}/sync_data_{seed}.csv'
        if not os.path.exists(sync_path):
            print(f'{sync_path} NOT FOUND')
            continue

        print(f'reading {sync_path}')
        df_sync_post = pd.read_csv(sync_path)
        res = ml_fn(df_sync_post, seed=0)
        res = res[res['Eval Data'] == 'Test']
        # res = res[res['Metric'] == 'f1_macro']
        print('seed=', seed, 'eps=', eps)
        print(res)
        for i, row in res.iterrows():
            target = row['target']
            metric = row['Metric']
            score = row['Score']
            Res.append([dataset_name, 'Yes', f'GSD', q_short_name, 'oneshot', 'oneshot', model, target, eps, metric, seed, score])
            # Res.append([dataset_name, 'Yes', f'GSD', q_short_name, 'LR', target, eps, 'Accuracy', seed, acc])

    results = pd.DataFrame(Res, columns=['Data', 'Is DP',
                                         'Generator',
                                         'Statistics',
                                         'T', 'S',
                                         'Model',
                                         'Target', 'epsilon', 'Metric',
                                         'seed',
                                         'Score'])

    print(results)
    os.makedirs('results_final', exist_ok=True)
    file_path = f'results_final/results_gsd_oneshot_2bt_{model}.csv'
    # file_path = f'results/results_gsd_ada_hs_{model}.csv'
    # if os.path.exists(file_path):
    #     results_pre = pd.read_csv(file_path, index_col=None)
    #     results = results_pre.append(results)
    print(f'Saving ', file_path)
    results.to_csv(file_path, index=False)