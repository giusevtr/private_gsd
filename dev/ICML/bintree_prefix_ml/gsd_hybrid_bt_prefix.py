import itertools
import jax.random
import pandas as pd
import os
from models import PrivGA, PrivGASparse
from stats import ChainedStatistics, Prefix, Marginals
import jax.numpy as jnp
from utils import timer, Dataset, Domain
from dp_data import load_domain_config, load_df, ml_eval


if __name__ == "__main__":
    module_name = 'BT+Prefix'
    # EPSILON = [0.07, 0.23, 0.52, 0.74, 1]
    EPSILON = [10]
    SEEDS = list(range(3))
    MAX_QUERIES = 400000
    dataset_name = 'folktables_2018_multitask_CA'

    PARAMS = [
        # (10, 50),       # 500
        (1, 50),        # 250
    ]

    os.makedirs('../prefix/icml_results/', exist_ok=True)
    file_name = '../prefix/icml_results/gsd_adaptive_prefix.csv'

    results_last = None
    if os.path.exists(file_name):
        print(f'reading {file_name}')
        results_last = pd.read_csv(file_name)

    Res = []

    root_path = '../../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)
    df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
    df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')

    domain = Domain.fromdict(config)
    data = Dataset(df_train, domain)

    targets = ['JWMNP_bin', 'PINCP', 'MIG', 'PUBCOV', 'ESR']
    features = []
    for f in domain.attrs:
        if f not in targets:
            features.append(f)

    model = 'LogisticRegression'
    ml_fn = ml_eval.get_evaluate_ml(df_test, config, targets, models=[model])

    cat_features = [(feat,) for feat in targets]
    binary_size = sum([domain.size(feat) for feat in cat_features])
    num_random_prefixes = MAX_QUERIES // binary_size
    module_bt = Marginals.get_all_kway_combinations(domain, k=2, bins=[2, 4, 8, 14, 32],
                                                    max_size=5000)
    module = Prefix(domain,
                    k_cat=1,
                    cat_kway_combinations=cat_features,
                    # cat_kway_combinations=[],
                    k_prefix=2,
                    num_random_prefixes=num_random_prefixes,
                    rng=jax.random.PRNGKey(0))
    stat_module = ChainedStatistics([module_bt, module])
    stat_module.fit(data)
    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module.get_dataset_statistics_fn()

    print(f'{dataset_name} has {len(domain.get_numeric_cols())} real features and '
          f'{len(domain.get_categorical_cols())} cat features.')
    print(f'Data cardinality is {domain.size()}.')
    print(f'Number of queries is {true_stats.shape[0]}.')
    print(f'train size: {df_train.shape}')
    print(f'test size:  {df_test.shape}')

    algo = PrivGA(num_generations=500000, domain=domain, data_size=2000, population_size=100, print_progress=False)
    delta = 1.0 / len(data) ** 2
    for eps, seed, (samples, epochs) in itertools.product(EPSILON, SEEDS, PARAMS):
        key = jax.random.PRNGKey(seed)
        t0 = timer()
        sync_dir = f'sync_data/{dataset_name}/GSD/{module_name}/{epochs}/{samples}/{eps:.2f}/'
        os.makedirs(sync_dir, exist_ok=True)
        sync_data = algo.fit_dp_hybrid(key, stat_module=stat_module,
                                        epsilon=eps, delta=delta,
                                         rounds=epochs, num_sample=samples,
                                         print_progress=True
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

        res = ml_fn(sync_data.df, seed=0)
        res = res[res['Eval Data'] == 'Test']
        res = res[res['Metric'] == 'f1_macro']
        print('seed=', seed, 'eps=', eps)
        print(res)

        # print('Saving', file_name)
        columns = ['Generator', 'Data', 'Statistics', 'T', 'S', 'epsilon', 'seed', 'error type', 'error', 'time']
        results_df = pd.DataFrame(Res, columns=columns)
        if results_last is not None:
            results_df = pd.concat([results_last, results_df], ignore_index=True)
        results_df.to_csv(file_name, index=False)





