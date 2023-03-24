import itertools
import jax.random
import pandas as pd
import os
from models import PrivGA, SimpleGAforSyncData
from stats import ChainedStatistics, Prefix
import jax.numpy as jnp
from utils import timer, Dataset, Domain , get_Xy, filter_outliers
from dp_data import load_domain_config, load_df


if __name__ == "__main__":
    module_name = 'Prefix'
    EPSILON = [0.07, 0.23, 0.52, 0.74, 1]
    SEEDS = list(range(3))
    MAX_QUERIES = 200000
    DATA = [
        # 'folktables_2018_real_CA',
        # 'folktables_2018_coverage_CA',
        'folktables_2018_employment_CA',
        'folktables_2018_income_CA',
        'folktables_2018_mobility_CA',
        'folktables_2018_travel_CA',
    ]

    PARAMS = [
        (200, 50),   # 10,000
        (150, 50),   # 7,500
        (100, 50),   # 5,000
        (40, 50),    # 2,000
        (20, 50),    # 1,000
        (10, 50),       # 500
        (5, 50),        # 250
    ]

    os.makedirs('icml_results/', exist_ok=True)
    file_name = 'icml_results/gsd_adaptive_prefix.csv'

    results_last = None
    if os.path.exists(file_name):
        print(f'reading {file_name}')
        results_last = pd.read_csv(file_name)

    Res = []
    for dataset_name in DATA:

        root_path = '../../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
        config = load_domain_config(dataset_name, root_path=root_path)
        df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
        df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')


        domain = Domain.fromdict(config)
        data = Dataset(df_train, domain)

        binary_features = [(feat,) for feat in domain.get_categorical_cols() if domain.size(feat)==2]
        binary_size = sum([domain.size(feat) for feat in binary_features])
        num_random_prefixes = MAX_QUERIES // binary_size
        module = Prefix(domain,
                        k_cat=1,
                        cat_kway_combinations=binary_features,
                        k_prefix=2,
                        num_random_prefixes=num_random_prefixes,
                        rng=jax.random.PRNGKey(0))
        stat_module = ChainedStatistics([module])
        stat_module.fit(data)
        true_stats = stat_module.get_all_true_statistics()
        stat_fn = stat_module.get_dataset_statistics_fn()

        print(f'{dataset_name} has {len(domain.get_numeric_cols())} real features and '
              f'{len(domain.get_categorical_cols())} cat features.')
        print(f'Data cardinality is {domain.size()}.')
        print(f'Number of queries is {true_stats.shape[0]}.')
        print(f'train size: {df_train.shape}')
        print(f'test size:  {df_test.shape}')

        algo = PrivGA(num_generations=100000,
                      strategy=SimpleGAforSyncData(domain, 2000), )
        delta = 1.0 / len(data) ** 2
        for eps, seed, (samples, epochs) in itertools.product(EPSILON, SEEDS, PARAMS):
            key = jax.random.PRNGKey(seed)
            t0 = timer()
            sync_dir = f'sync_data/{dataset_name}/GSD/{module_name}/{epochs}/{samples}/{eps:.2f}/'
            os.makedirs(sync_dir, exist_ok=True)
            sync_data = algo.fit_dp_adaptive(key, stat_module=stat_module,
                                    epsilon=eps, delta=delta,
                                             rounds=epochs, num_sample=samples,
                                             print_progress=False
                                    )
            sync_data.df.to_csv(f'{sync_dir}/sync_data_{seed}.csv', index=False)
            errors = jnp.abs(true_stats - stat_fn(sync_data))
            elapsed_time = timer() - t0
            print(f'GSD({dataset_name, module_name}): eps={eps:.2f}, seed={seed}'
                  f'\t max error = {errors.max():.5f}'
                  f'\t avg error = {errors.mean():.5f}'
                  f'\t time = {elapsed_time:.4f}')
            Res.append(
                ['GSD', dataset_name, module_name, epochs, samples, eps, seed, 'Max', errors.max(), elapsed_time])
            Res.append(['GSD', dataset_name, module_name, epochs, samples, eps, seed, 'Average', errors.mean(),
                        elapsed_time])

            # print('Saving', file_name)
            columns = ['Generator', 'Data', 'Statistics', 'T', 'S', 'epsilon', 'seed', 'error type', 'error', 'time']
            results_df = pd.DataFrame(Res, columns=columns)
            if results_last is not None:
                results_df = pd.concat([results_last, results_df], ignore_index=True)
            results_df.to_csv(file_name, index=False)





