import os, sys
import jax.random
import pandas as pd
import numpy as np
from models import GeneticSDConsistent as GeneticSD
from models import GSD
from stats import ChainedStatistics,  Marginals, NullCounts
import jax.numpy as jnp
from dp_data import load_domain_config, load_df
from eval_ml import  get_Xy, get_evaluate_ml
from dp_data.data_preprocessor import DataPreprocessor
from utils import timer, Dataset, Domain
import sklearn

import matplotlib.pyplot as plt

from sklearn.metrics import make_scorer, f1_score, roc_auc_score, average_precision_score, accuracy_score, recall_score, precision_score
from catboost import CatBoostClassifier, Pool

cat_only = True
eps = 100000
# data_size_str = '32000'
# data_size_str = '32000'
data_size_str = '2000'
k = 3
QUANTILES = 30
DATA = [
    # ('folktables_2018_coverage_CA', 'PUBCOV'),
    # ('folktables_2018_mobility_CA', 'MIG'),
    # ('folktables_2018_employment_CA', 'ESR'),
    # ('folktables_2018_income_CA', 'PINCP'),
    ('folktables_2018_travel_CA', 'JWMNP'),
]

seed = 0
print('Cat only ? ', cat_only)
print(f'seed={seed}')
for dataset_name, target in DATA:
    root_path = 'dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)
    domain = Domain(config=config)
    df_train = load_df(dataset_name, root_path=root_path, idxs_path=f'seed{seed}/train')
    df_test = load_df(dataset_name, root_path=root_path, idxs_path=f'seed{seed}/test')

    df_train_sample = df_train.sample(n=1000)

    data_size = int(data_size_str)
    sync_dir = f'sync_data/{dataset_name}/{k}/{eps:.2f}/{data_size_str}/oneshot'
    print(f'reading ', f'{sync_dir}/sync_data_{seed}.csv')
    df_sync = pd.read_csv(f'{sync_dir}/sync_data_{seed}.csv')

    if cat_only:
        cat_cols = domain.get_categorical_cols() + domain.get_ordinal_cols()
        domain = domain.project(cat_cols)

    features = list(domain.attrs)
    features.remove(target)

    df_train_X = df_train[features].astype(int)
    df_train_y = df_train[[target]].astype(int)

    df_train_sample_X = df_train_sample[features].astype(int)
    df_train_sample_y = df_train_sample[[target]].astype(int)

    df_test_X = df_test[features].astype(int)
    df_test_y = df_test[[target]].astype(int)

    df_sync_X = df_sync[features].astype(int)
    df_sync_y = df_sync[[target]].astype(int)

    cat_feats = domain.get_categorical_cols()[:-1]
    # cat_feats = 'JWTR', 'POWPUMA', 'SEX', 'ESP', 'CIT', 'MAR', 'OCCP', 'RELP', 'PUMA', 'DIS', 'RAC1P', 'MIG', 'SCHL'
    print(cat_feats)

    scorer = make_scorer(f1_score, average='macro')

    # Train on Original
    print(f'REAL:')
    model = CatBoostClassifier(task_type="GPU", random_seed=0, verbose=False)
    model.fit(df_train_X, df_train_y, cat_features=cat_feats)
    pred = model.predict(df_test_X)
    f1 = f1_score(df_test_y.values, pred, average='macro')
    recall = recall_score(df_test_y.values, pred)
    precision = precision_score(df_test_y.values, pred)
    print(f'\tf1={f1}, recal={recall:.4f}, precision={precision:.4f} ')

    print(f'REAL-sample:')
    model = CatBoostClassifier(task_type="GPU", random_seed=0, verbose=False)
    model.fit(df_train_sample_X, df_train_sample_y, cat_features=cat_feats)
    pred = model.predict(df_test_X)
    f1 = f1_score(df_test_y.values, pred, average='macro')
    recall = recall_score(df_test_y.values, pred)
    precision = precision_score(df_test_y.values, pred)
    print(f'\tf1={f1}, recal={recall:.4f}, precision={precision:.4f} ')

    # Train on Sync
    print(f'SYNC:')
    model = CatBoostClassifier(task_type="GPU", random_seed=0, verbose=False)
    model.fit(df_sync_X, df_sync_y, cat_features=cat_feats)
    pred = model.predict(df_test_X)
    f1 = f1_score(df_test_y.values, pred, average='macro')
    recall = recall_score(df_test_y.values, pred)
    precision = precision_score(df_test_y.values, pred)
    print(f'\tf1={f1}, recal={recall:.4f}, precision={precision:.4f} ')


