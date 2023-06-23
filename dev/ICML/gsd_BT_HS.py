import itertools

import jax.random
import matplotlib.pyplot as plt
import pandas as pd
import os
from models import GSD, SimpleGAforSyncData, PrivGAJit
from stats import ChainedStatistics, Marginals, HalfspacesBT
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


def run(dataset_name, target,  seeds=(0, 1, 2), eps_values=(0.07, 0.23, 0.52, 0.74, 1.0)):
    module_name = 'BT+HS'
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
    # module0 = MarginalsDiff.get_all_kway_categorical_combinations(data.domain, k=2)


    targets = [target]
    # targets = ['PINCP', 'PUBCOV']
    features = []
    for f in domain.attrs:
        if f not in targets:
            features.append(f)
    model = 'LogisticRegression'
    ml_fn = ml_eval.get_evaluate_ml(df_test, config, targets, models=[model])


    # module = Marginals.get_all_kway_combinations(domain, k=2, bins=[2, 4, 8, 16, 32])
    module = HalfspacesBT(data.domain, key=jax.random.PRNGKey(0), random_proj=10000, bins=(2, 4, 8, 16, 32))


    stat_module = ChainedStatistics([module])
    stat_module.fit(data)
    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module.get_dataset_statistics_fn()

    print(f'{dataset_name} has {len(domain.get_numeric_cols())} real features and '
          f'{len(domain.get_categorical_cols())} cat features.')
    print(f'Data cardinality is {domain.size()}.')
    print(f'Number of queries is {true_stats.shape[0]}.')

    algo = GSD(num_generations=20000,
               domain=domain, data_size=1000, population_size=100, muta_rate=1, mate_rate=1,
               print_progress=True)

    delta = 1.0 / len(data) ** 2
    for seed in seeds:
        for eps in eps_values:
            key = jax.random.PRNGKey(seed)
            t0 = timer()
            sync_dir = f'sync_data/{dataset_name}/GSD/{module_name}/oneshot/oneshot/{eps:.2f}/'
            os.makedirs(sync_dir, exist_ok=True)
            sync_data = algo.fit_dp_adaptive(key, stat_module=stat_module,
                                           epsilon=eps, delta=delta,
                                           # oneshot_share_opt=0.9,
                                           rounds=50
                                           )



            # sync_data.df.to_csv(f'{sync_dir}/sync_data_{seed}.csv', index=False)
            errors = jnp.abs(true_stats - stat_fn(sync_data))
            elapsed_time = timer() - t0
            print(f'GSD({dataset_name, module_name}): eps={eps:.2f}, seed={seed}'
                  f'\t max error = {errors.max():.5f}'
                  f'\t avg error = {errors.mean():.5f}'
                  f'\t time = {elapsed_time:.4f}')
            Res.append(['GSD', dataset_name, module_name, "oneshot", "oneshot", eps, seed, 'Max', errors.max(), elapsed_time])
            Res.append(['GSD', dataset_name, module_name, "oneshot", "oneshot", eps, seed, 'Average', errors.mean(), elapsed_time])


            res = ml_fn(sync_data.df, seed=0)
            res = res[res['Eval Data'] == 'Test']
            res = res[res['Metric'] == 'f1_macro']
            print('seed=', seed, 'eps=', eps)
            print(res)
            for i, row in res.iterrows():
                target = row['target']
                f1 = row['Score']
                print(f'target={target}, f1={f1:.5f}')


        print()

    columns = ['Generator', 'Data', 'Statistics', 'T', 'S', 'epsilon', 'seed', 'error type', 'error', 'time']
    results_df = pd.DataFrame(Res, columns=columns)
    return results_df

if __name__ == "__main__":

    DATA = [
        ('folktables_2018_coverage_CA', 'PUBCOV'),
        # 'folktables_2018_employment_CA',
        # 'folktables_2018_income_CA',
        # 'folktables_2018_mobility_CA',
        # 'folktables_2018_travel_CA',
    ]

    os.makedirs('icml_results/', exist_ok=True)
    file_name = 'icml_results/oneshot_ranges/gsd_oneshot.csv'
    results = None
    if os.path.exists(file_name):
        print(f'reading {file_name}')
        results = pd.read_csv(file_name)
    for data , target in DATA:
        results_temp = run(data, target, eps_values=[1.0], seeds=[0])
        results = pd.concat([results, results_temp], ignore_index=True) if results is not None else results_temp
        print(f'Saving: {file_name}')
        # results.to_csv(file_name, index=False)

