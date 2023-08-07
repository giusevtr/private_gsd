import pandas as pd
import os
from models import GSD
from utils import Dataset, Domain
import numpy as np

import jax.random
from models import GeneticSDConsistent as GeneticSD
from models import GSD
from stats import ChainedStatistics,  Marginals, NullCounts
import jax.numpy as jnp

from stats.get_marginals_fn import get_marginal_query
QUANTILES = 50
k = 2
from dp_data import cleanup, DataPreprocessor, get_config_from_json


for seed in range(5):

    data_path = '../data2/data'
    X_train = np.load(f'{data_path}/diabetes/kfolds/{seed}/X_num_train.npy')
    X_test = np.load( f'{data_path}/diabetes/kfolds/{seed}/X_num_test.npy')
    X_val = np.load(  f'{data_path}/diabetes/kfolds/{seed}/X_num_val.npy')
    y_train = np.load(f'{data_path}/diabetes/kfolds/{seed}/y_train.npy')
    y_test = np.load( f'{data_path}/diabetes/kfolds/{seed}/y_test.npy')
    y_val = np.load(  f'{data_path}/diabetes/kfolds/{seed}/y_val.npy')

    all_cols = ['Pregnant',
               'Plasma',
               'BloodPressure',  # 122
               'Triceps',  # 99
               'Insulin',  # 846
               'BMI',  # 67.1
               'Diabetes_pedigree',  # 2.342
               'Age',  # 60
               'Label']

    train_df = pd.DataFrame(np.column_stack((X_train, y_train)), columns=all_cols)
    val_df = pd.DataFrame(np.column_stack((X_val, y_val)), columns=all_cols)
    test_df = pd.DataFrame(np.column_stack((X_test, y_test)), columns=all_cols)
    all_df = pd.concat([train_df, val_df, test_df])

    X = np.concatenate((X_train, X_test, X_val))
    y = np.concatenate((y_train, y_test, y_val))

    ordinal = ['Pregnant',  # 17
               'Plasma',  # 199
               'BloodPressure',  # 122
               'Triceps', 'Age']
    real_cols = [
        'Insulin',  # 846
        'BMI',  # 67.1
        'Diabetes_pedigree',  # 2.342
    ]
    cat_cols = ['Label']


    config = get_config_from_json({'categorical': cat_cols , 'ordinal': ordinal, 'numerical': real_cols})
    preprocessor = DataPreprocessor(config=config)
    preprocessor.fit(all_df)
    pre_train_df = preprocessor.transform(train_df)
    pre_val_df = preprocessor.transform(val_df)
    pre_test_df = preprocessor.transform(test_df)



    # df_pre = pd.DataFrame(np.column_stack((X_pre, y_train)), columns=columns)
    domain = Domain(preprocessor.get_domain())
    data = Dataset(pre_train_df, domain)
    print()
    true_stats, stat_fn = get_marginal_query(data, domain, k=k, min_bin_density=0.005,
                                             min_marginal_size=600,
                                             include_features=['Label'], verbose=True)

    print(f'Debug: ', jnp.linalg.norm(stat_fn(data.to_numpy())-true_stats))
    key = jax.random.PRNGKey(seed)
    N = len(data.df)
    algo = GSD(num_generations=600000,
                       print_progress=False,
                       stop_early=True,
                       domain=data.domain,
                       population_size=50,
                       data_size=N,
                       stop_early_gen=N,
                       )
    # delta = 1.0 / len(data) ** 2
    sync_data = algo.fit_help(key, true_stats, stat_fn)
    sync_data_df = sync_data.df.copy()
    sync_data_df_post = preprocessor.inverse_transform(sync_data_df)

    dataset_name = 'diabetes'
    data_size_str = 'N'
    sync_dir = f'sync_data/{dataset_name}/{k}/inf/{data_size_str}/oneshot'
    os.makedirs(sync_dir, exist_ok=True)
    print(f'Saving {sync_dir}/sync_data_{seed}.csv')
    sync_data_df_post.to_csv(f'{sync_dir}/sync_data_{seed}.csv', index=False)

    errors = jnp.abs(true_stats - stat_fn(sync_data.to_numpy()))

    print(f'Input data {dataset_name}, k={k}, seed={seed}')
    print(f'GSD(oneshot):'
          f'\t max error = {errors.max():.5f}'
          f'\t avg error = {errors.mean():.6f}')

    ## TODO: add marginal error over all queries.
    ## TODO: Preserve all 1-way marginals.