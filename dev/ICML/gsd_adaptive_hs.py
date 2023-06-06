import itertools
import sys

import jax.random
import matplotlib.pyplot as plt
import pandas as pd
import os
from models import GSD, PrivGAJit
from stats import ChainedStatistics, Marginals, HalfspacesPrefix
# from utils.utils_data import get_data
from utils import timer
import jax.numpy as jnp
# from dp_data.data import get_data
from utils import timer, Dataset, Domain , get_Xy, filter_outliers
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression

from dp_data import load_domain_config, load_df, ml_eval

from dev.dataloading.data_functions.acs import get_acs_all


def run(dataset_name,
        rounds, num_sample,
        max_num_queries=1000000,
        seeds=(0, 1, 2), eps_values=(0.07, 0.23, 0.52, 0.74, 1.0),
        ):
    module_name = 'HS'
    Res = []

    root_path = '../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)
    df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
    df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')

    print(f'train size: {df_train.shape}')
    print(f'test size:  {df_test.shape}')
    domain = Domain.fromdict(config)
    data = Dataset(df_train, domain)




    # Create statistics and evaluate
    # module0 = Marginals.get_kway_categorical(data.domain, k=2)


    module1 = HalfspacesPrefix(data.domain,
                        key=jax.random.PRNGKey(0),
                        random_proj=max_num_queries)




    print(f'{dataset_name} has {len(domain.get_numeric_cols())} real features and '
          f'{len(domain.get_categorical_cols())} cat features.')
    print(f'Data cardinality is {domain.size()}.')

    stat_module = ChainedStatistics([module1])
    stat_module.fit(data)
    stat_fn = stat_module.get_dataset_statistics_fn()

    delta = 1.0 / len(data) ** 2
    for seed in seeds:
        for eps in eps_values:

            print(f'Number of queries is {stat_module.get_all_true_statistics().shape[0]}.')
            algo = GSD(num_generations=200000,
                       domain=domain, data_size=2000, population_size=100, muta_rate=1, mate_rate=1,
                       print_progress=False)

            key = jax.random.PRNGKey(seed)
            t0 = timer()
            sync_dir = f'sync_data/{dataset_name}/GSD/{module_name}/{rounds}/{num_sample}/{eps:.2f}/'
            os.makedirs(sync_dir, exist_ok=True)
            sync_data = algo.fit_dp_adaptive(key, stat_module=stat_module,
                                           epsilon=eps, delta=delta,
                                           rounds=rounds,
                                           num_sample=num_sample,
                                             print_progress=False
                                           )
            sync_data.df.to_csv(f'{sync_dir}/sync_data_{seed}.csv', index=False)
            errors = jnp.abs(stat_module.get_all_true_statistics() - stat_fn(sync_data))
            elapsed_time = timer() - t0
            print(f'GSD({dataset_name, module_name}): eps={eps:.2f}, seed={seed}'
                  f'\t max error = {errors.max():.5f}'
                  f'\t avg error = {errors.mean():.5f}'
                  f'\t time = {elapsed_time:.4f}')
            Res.append(['GSD', dataset_name, module_name, rounds, num_sample, eps, seed, 'Max', errors.max(), elapsed_time])
            Res.append(['GSD', dataset_name, module_name, rounds, num_sample, eps, seed, 'Average', errors.mean(), elapsed_time])

        print()

    columns = ['Generator', 'Data', 'Statistics', 'T', 'S', 'epsilon', 'seed', 'error type', 'error', 'time']
    results_df = pd.DataFrame(Res, columns=columns)
    return results_df

if __name__ == "__main__":

    save_path = sys.argv[1]
    data_name = sys.argv[2]
    print(f'Save path={save_path}')

    file_name = save_path
    # os.makedirs(file_name, exist_ok=True)
    results = None
    if os.path.exists(file_name):
        print(f'reading {file_name}')
        results = pd.read_csv(file_name)
    # for data in DATA:
    results_temp = run(data_name,
                       rounds=int(sys.argv[3]),
                       num_sample=int(sys.argv[4]),
                       max_num_queries=200000,
                       eps_values=[float(sys.argv[5])],
                       seeds=[int(sys.argv[6])])
    # results_temp = run(data, eps_values=[1.0], seeds=[0])
    results = pd.concat([results, results_temp], ignore_index=True) if results is not None else results_temp
    print(f'Saving: {file_name}')
    results.to_csv(file_name, index=False)

