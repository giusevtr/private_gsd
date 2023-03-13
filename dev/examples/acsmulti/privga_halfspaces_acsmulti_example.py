import os

import jax.random
import matplotlib.pyplot as plt
import pandas as pd

from models import PrivGA, SimpleGAforSyncData, RelaxedProjectionPP
from stats import ChainedStatistics, Halfspace, Prefix, Marginals
# from utils.utils_data import get_data
from utils import timer
import jax.numpy as jnp
# from dp_data.data import get_data
from utils import timer, Dataset, Domain, filter_outliers
import seaborn as sns

from dev.dataloading.data_functions.acs import get_acs_all

from sklearn.linear_model import LogisticRegression
from dev.ML.ml_utils import filter_outliers, evaluate_machine_learning_task

if __name__ == "__main__":
    epsilon_vals = [0.07, 0.23, 0.52, 0.74, 1, 10]
    epsilon_vals.reverse()
    # epsilon_vals = [1, 10]
    dataset_name = 'folktables_2018_multitask_NY'

    data_container_fn = get_acs_all(state='NY')
    data_container = data_container_fn(seed=0)

    domain = data_container.train.domain
    cat_cols = domain.get_categorical_cols()
    num_cols = domain.get_numeric_cols()
    target = 'PINCP'
    targets = ['PINCP',  'PUBCOV', 'ESR', 'MIG', 'JWMNP' ]
    features = []
    for f in domain.attrs:
        if f not in targets:
            features.append(f)


    # df_train = data_container.from_dataset_to_df_fn(
    #     data_container.train
    # )
    df_test = data_container.from_dataset_to_df_fn(
        data_container.test
    )
    data = data_container.train

    #############
    ## ML Function

    # Create statistics and evaluate

    key = jax.random.PRNGKey(0)
    module0 = Marginals.get_all_kway_combinations(data.domain, k=2, bins=[2, 4, 8, 16, 32])
    module1 = Halfspace(domain=data.domain, k_cat=1,
                        cat_kway_combinations=[(tar,) for tar in targets], rng=key,
                        num_random_halfspaces=10000)
    stat_module = ChainedStatistics([module0,
                                     module1
                                     ])
    stat_module.fit(data)


    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module._get_workload_fn()

    algo = PrivGA(num_generations=40000, print_progress=False, stop_early=True, strategy=SimpleGAforSyncData(domain=data.domain, elite_size=2, data_size=1000))
    # Choose algorithm parameters

    delta = 1.0 / len(data) ** 2
    # Generate differentially private synthetic data with ADAPTIVE mechanism
    # for eps in [1]:
    for eps in epsilon_vals:
        for seed in [0, 1, 2]:
        # for seed in [0]:
            sync_dir = f'sync_data/{dataset_name}/PrivGA/Ranges/oneshot/{eps:.2f}/'
            sync_dir_post = f'sync_data_post/{dataset_name}/PrivGA/Ranges/oneshot/{eps:.2f}/'
            os.makedirs(sync_dir, exist_ok=True)
            os.makedirs(sync_dir_post, exist_ok=True)

            key = jax.random.PRNGKey(seed)
            t0 = timer()

            def debug(i, sync_data: Dataset):
                print(f'results epoch {i}:')
                clf = LogisticRegression(max_iter=5000, random_state=seed,
                                         solver='liblinear', penalty='l1')
                df_train = data_container.from_dataset_to_df_fn(sync_data)
                ml_result = evaluate_machine_learning_task(df_train, df_test,
                                                           feature_columns=features,
                                                           label_column=target,
                                                           cat_columns=cat_cols,
                                                           num_columns=num_cols,
                                                           endmodel=clf)
                print(ml_result)




            sync_data = algo.fit_dp(key, stat_module=stat_module, epsilon=eps, delta=delta)

            sync_data.df.to_csv(f'{sync_dir}/sync_data_{seed}.csv', index=False)
            errors = jnp.abs(true_stats - stat_fn(sync_data.to_numpy()))
            print(f'PrivGA(oneshot): eps={eps:.2f}, seed={seed}'
                  f'\t max error = {errors.max():.5f}'
                  f'\t avg error = {errors.mean():.5f}'
                  f'\t time = {timer() - t0:.4f}')
            print('Final ML Results:')
            debug(-1, sync_data)

        print()

