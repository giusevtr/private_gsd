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

cat_only = True
eps = 100000
# data_size_str = '32000'
data_size_str = '32000'
k = 3
SEEDS = [0]
QUANTILES = 30
DATA = [

    ('folktables_2018_coverage_CA', 'PUBCOV'),
    ('folktables_2018_mobility_CA', 'MIG'),
    ('folktables_2018_employment_CA', 'ESR'),
    ('folktables_2018_income_CA', 'PINCP'),
    # ('folktables_2018_travel_CA', 'JWMNP'),

]

print('Cat only ? ', cat_only)
seed = 0
for dataset_name, target in DATA:
    root_path = 'dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)
    domain = Domain(config=config)
    df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
    df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')

    data_size = int(data_size_str)
    sync_dir = f'sync_data/{dataset_name}/{k}/{eps:.2f}/{data_size_str}/oneshot'
    print(f'reading {sync_dir}')
    df_sync = pd.read_csv(f'{sync_dir}/sync_data_{seed}.csv')

    features = df_sync.columns

    if cat_only:
        cat_cols = domain.get_categorical_cols() + domain.get_ordinal_cols()
        domain = domain.project(cat_cols)
    eval_ml = get_evaluate_ml(
                              domain=domain,
                                targets=[target],
                              # models=['GradientBoosting'],
                            # models=['RandomForest'],
                                models=[
                                    'DecisionTree',
                                    # 'KNN',
                                    'LogisticRegression',
                                    'LinearSVC',
                                    'RandomForest',
                                    'AdaBoost',
                                    'GradientBoosting',
                                    'XGBoost',
                                    'LightGBM',
                                ],
                              grid_search=False
                              )


    print(f'DATA={dataset_name}')


    print('SYNC:')
    sync_df = eval_ml(df_sync, df_test, 0, group=None, verbose=True)
    sync_df = sync_df[(sync_df['Eval Data'] == 'Test') & (sync_df['Metric'] == 'f1_macro')]
    print(sync_df)

    print('REAL:')
    real_df = eval_ml(df_train, df_test, 0, group=None, verbose=True)
    real_df = real_df[(real_df['Eval Data'] == 'Test') & (real_df['Metric'] == 'f1_macro')]
    print(real_df)
    print('\n')

    sync_df['Type'] = 'Sync'
    real_df['Type'] = 'Real'
    df = pd.concat((sync_df, real_df))
    sns.barplot(data=df, x='Model', y='Score', hue='Type')
    plt.xticks(rotation=25)
    plt.title(f'{dataset_name} 32k, Cond 3-way Marginals')
    plt.show()

