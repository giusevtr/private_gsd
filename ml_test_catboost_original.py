import itertools
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
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import make_scorer, f1_score, roc_auc_score, average_precision_score, accuracy_score

from catboost import CatBoostClassifier, Pool


eps = 100000
# data_size_str = '32000'
COLUMNS = [
    'Model', 'target', 'Eval Data', 'Metric', 'Score', 'Sub Score', 'Data', 'Type', 'N', 'Categorical Only', 'Seed'
]
data_size_str_list = ['2000', '4000', '8000', '16000', '32000']
k = 3
SEEDS = [0]
QUANTILES = 30
DATA = [
    ('folktables_2018_coverage_CA', 'PUBCOV'),
    ('folktables_2018_mobility_CA', 'MIG'),
    ('folktables_2018_employment_CA', 'ESR'),
    ('folktables_2018_income_CA', 'PINCP'),
    ('folktables_2018_travel_CA', 'JWMNP'),
]

Results = []
for (dataset_name, target), seed, cat_only in itertools.product(DATA, SEEDS, [True, False]):
    root_path = 'dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)
    domain = Domain(config=config)
    df_train = load_df(dataset_name, root_path=root_path, idxs_path=f'seed{seed}/train')
    df_test = load_df(dataset_name, root_path=root_path, idxs_path=f'seed{seed}/test')

    if cat_only:
        cat_cols = domain.get_categorical_cols() + domain.get_ordinal_cols()
        domain = domain.project(cat_cols)

    cat_cols = domain.get_categorical_cols() + domain.get_ordinal_cols()
    if cat_only:
        domain = domain.project(cat_cols)
    if target in cat_cols:
        cat_cols.remove(target)

    features = list(domain.attrs)
    features.remove(target)

    df_test_X = df_test[features]
    df_test_X[cat_cols] = df_test[cat_cols].astype(int)
    df_test_y = df_test[[target]].astype(int)

    df_train_X = df_train[features]
    df_train_X[cat_cols] = df_train[cat_cols].astype(int)
    df_train_y = df_train[[target]].astype(int)

    model = CatBoostClassifier(task_type="GPU", random_seed=0, verbose=False)
    model.fit(df_train_X, df_train_y, cat_features=cat_cols)
    pred = model.predict(df_test_X)
    f1_test = f1_score(df_test_y.values, pred, average='macro')
    acc_test = accuracy_score(df_test_y.values, pred)

    f1_train = f1_score(df_train_y.values, model.predict(df_train_X), average='macro')
    acc_train = accuracy_score(df_train_y.values, model.predict(df_train_X))
    print(f'{dataset_name}, cat={cat_only}, seed={seed}: f1={f1_test}, acc={acc_test}')

    # 'Model', 'target', 'Eval Data', 'Metric', 'Score', 'Sub Score', 'Data', 'Type', 'N', 'Categorical Only', 'Seed'
    res1 = ['Catboost', target, 'Test', 'f1_macro', f1_test, None, dataset_name,    'Original', 'N', cat_only, seed]
    res2 = ['Catboost', target, 'Train', 'f1_macro', f1_train, None, dataset_name,  'Original', 'N', cat_only, seed]
    res3 = ['Catboost', target, 'Test', 'accuracy', acc_test, None, dataset_name,   'Original', 'N', cat_only, seed]
    res4 = ['Catboost', target, 'Train', 'accuracy', acc_train, None, dataset_name, 'Original', 'N', cat_only, seed]
    result_df = pd.DataFrame([res1, res2, res3, res4], columns=COLUMNS)

    Results.append(result_df)


results_df = pd.concat(Results)
results_df.to_csv('results/acs_original_catboost_results.csv', index=False)

