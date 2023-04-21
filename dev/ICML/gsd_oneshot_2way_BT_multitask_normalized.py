import itertools
import numpy as np
import jax.random
import matplotlib.pyplot as plt
import pandas as pd
import os
from models import PrivGAV2, SimpleGAforSyncData, PrivGAJit
from stats import ChainedStatistics, Marginals
import jax.numpy as jnp
from utils import timer, Dataset, Domain
from dp_data import load_domain_config, load_df, ml_eval, DataPreprocessor
import pickle
from dev.dataloading.data_functions.acs import get_acs_all


def run(dataset_name,  seeds=(0, 1, 2), eps_values=(0.07, 0.23, 0.52, 0.74, 1.0),
        evaluate_only=False):
    module_name = '2-Mixed BT'

    root_path = '../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)
    df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
    df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')

    print(f'train size: {df_train.shape}')
    print(f'test size:  {df_test.shape}')
    domain = Domain.fromdict(config)

    for num_col in domain.get_numeric_cols():
        all_values = np.concatenate((df_train[num_col].values, df_test[num_col].values))
        std = all_values.std()
        df_train[num_col] = df_train[num_col] / std
        df_test[num_col] = df_test[num_col] / std

        all_values = np.concatenate((df_train[num_col].values, df_test[num_col].values))
        qu95 = np.quantile(all_values, 0.95)
        qu05 = np.quantile(all_values, 0.05)

        outliers95 = np.sum(df_train[num_col].values > qu95)
        outliers05 = np.sum(df_train[num_col].values < qu05)
        df_train[num_col] = np.clip(df_train[num_col], qu05, qu95)
        df_train[num_col] = (df_train[num_col].values - qu05) / (qu95 - qu05)

        df_test[num_col] = np.clip(df_test[num_col], qu05, qu95)
        df_test[num_col] = (df_test[num_col].values - qu05) / (qu95 - qu05)


        std_new = df_train[num_col].values.std()
        max_new = df_train[num_col].values.max()
        min_new = df_train[num_col].values.min()
        # outliers = np.sum(data.df[num_col].values > 1)
        print(f'{num_col:<5}: std={std:<3.4f}, std_new = {std_new:<3.4}, min_new= {min_new:<3.4}, max_new = {max_new:<3.4}, num outliers={outliers05 + outliers95:<3}')


    data = Dataset(df_train, domain)
    targets = ['JWMNP_bin', 'PINCP', 'MIG', 'PUBCOV', 'ESR']
    features = []
    for f in domain.attrs:
        if f not in targets:
            features.append(f)

    ml_fn = ml_eval.get_evaluate_ml(df_test, config, targets, models=['LogisticRegression', 'XGBoost'])
    orig_result = ml_fn(df_train, 0)

    orig_result = orig_result[orig_result['Eval Data'] == 'Test']
    print(orig_result)


    module = Marginals.get_all_kway_combinations(domain, k=2, bins=[2, 4, 8, 16, 32], max_size=20000)
    stat_module = ChainedStatistics([module])
    stat_module.fit(data)
    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module.get_dataset_statistics_fn()

    print(f'{dataset_name} has {len(domain.get_numeric_cols())} real features and '
          f'{len(domain.get_categorical_cols())} cat features.')
    print(f'Data cardinality is {domain.size()}.')
    print(f'Number of queries is {true_stats.shape[0]}.')

    algo = PrivGAV2(num_generations=100000,
                  domain=domain, data_size=2000, population_size=100, muta_rate=1, mate_rate=1,
                  print_progress=True)

    delta = 1.0 / len(data) ** 2

    Res = []
    ml_results = []
    for seed in seeds:
        for eps in eps_values:
            key = jax.random.PRNGKey(seed)
            t0 = timer()
            sync_dir = f'sync_data/{dataset_name}/GSD/{module_name}/oneshot/oneshot/{eps:.2f}/'
            os.makedirs(sync_dir, exist_ok=True)
            sync_data_path = f'{sync_dir}/sync_data_{seed}.csv'
            if evaluate_only:
                df = pd.read_csv(sync_data_path)
                sync_data = Dataset(df, domain)
            else:
                sync_data = algo.fit_dp(key, stat_module=stat_module,
                                           epsilon=eps, delta=delta,
                                           )
                # sync_data.df.to_csv(sync_data_path, index=False)

            errors = jnp.abs(true_stats - stat_fn(sync_data))
            elapsed_time = timer() - t0
            print(f'GSD({dataset_name, module_name}): eps={eps:.2f}, seed={seed}'
                  f'\t max error = {errors.max():.5f}'
                  f'\t avg error = {errors.mean():.5f}'
                  f'\t time = {elapsed_time:.4f}')
            Res.append(['GSD', dataset_name, module_name, "oneshot", "oneshot", eps, seed, 'Max', errors.max(), elapsed_time])
            Res.append(['GSD', dataset_name, module_name, "oneshot", "oneshot", eps, seed, 'Average', errors.mean(), elapsed_time])

            # sync_data_post = preprocesor.inverse_transform(sync_data.df)
            res = ml_fn(sync_data.df, seed=0)
            res = res[res['Eval Data'] == 'Test']
            print('seed=', seed, 'eps=', eps)
            print(res)
            for i, row in res.iterrows():
                model_name = row['Model']
                target = row['target']
                metric = row['Metric']
                score = row['Score']
                ml_results.append([dataset_name, 'Yes', f'GSD', 'BT', 'oneshot', 'oneshot', model_name, target, eps, metric, seed, score])

        print()

    columns = ['Generator', 'Data', 'Statistics', 'T', 'S', 'epsilon', 'seed', 'error type', 'error', 'time']
    results_df = pd.DataFrame(Res, columns=columns)


    ml_results_df = pd.DataFrame(ml_results, columns=['Data', 'Is DP', 'Generator',
                                         'Statistics',
                                         'T', 'S',
                                         'Model',
                                         'Target', 'epsilon', 'Metric',
                                         'seed',
                                         'Score'
                                        ])



    return results_df, ml_results_df

if __name__ == "__main__":

    DATA = [
        'folktables_2018_multitask_CA',
    ]

    os.makedirs('icml_results/', exist_ok=True)
    os.makedirs('icml_ml_results/', exist_ok=True)
    file_name = 'icml_results/gsd_oneshot_2way_BT.csv'
    results = None
    if os.path.exists(file_name):
        print(f'reading {file_name}')
        results = pd.read_csv(file_name)
    for data in DATA:
        results_temp, results_ml = run(data,
                           # eps_values=[1.0, 0.74, 0.52, 0.23, 0.07],
                           eps_values=[1],
                           seeds=[0],
                           evaluate_only=False)
        results = pd.concat([results, results_temp], ignore_index=True) if results is not None else results_temp
        print(f'Saving: {file_name}')
        # results.to_csv(file_name, index=False)

        ml_file_name = f'icml_ml_results/gsd_oneshot_2way_BT_{data}.csv'
        results.to_csv(file_name, index=False)


