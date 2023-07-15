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


eps = 10000
data_size_str = '16000'
k = 2
SEEDS = [0]
QUANTILES = 20
DATA = [
    'folktables_2018_coverage_CA',
    'folktables_2018_mobility_CA',
    # 'folktables_2018_employment_CA',
    # 'folktables_2018_income_CA',
    # 'folktables_2018_travel_CA',
]

seed = 0
for dataset_name in DATA:
    root_path = 'dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)
    df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
    df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')

    data_size = int(data_size_str)
    sync_dir = f'sync_data/{dataset_name}/{k}/{eps:.2f}/{data_size_str}/oneshot'
    df_sync = pd.read_csv(f'{sync_dir}/sync_data_{seed}.csv')
    df_real = df_train.sample(n=data_size)

    df_real['Label'] = 1
    df_sync['Label'] = 0

    features = df_sync.columns
    config['Label'] = {"type": "categorical", "size": 2}
    df = pd.concat((df_real, df_sync))

    eval_ml = get_evaluate_ml(df,
                                             config=config,
                                             targets=['Label'],
                                             # models=['LogisticRegression'],
                              models=['RandomForest'],
                              # models=['DecisionTree'],
                              )

    results_df = eval_ml(df, 0, verbose=True)
    print(f'DATA={dataset_name}')
    print(results_df)

