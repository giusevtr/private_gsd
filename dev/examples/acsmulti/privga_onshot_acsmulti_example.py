import os

import jax.random
import matplotlib.pyplot as plt
import pandas as pd

from models import GeneticSD, GeneticStrategy, RelaxedProjectionPP
from stats import ChainedStatistics, Halfspace, Prefix, Marginals
# from utils.utils_data import get_data
from utils import timer
import jax.numpy as jnp
# from dp_data.data import get_data
from dp_data import load_domain_config, load_df, get_evaluate_ml
from utils import timer, Dataset, Domain, filter_outliers
import seaborn as sns

if __name__ == "__main__":
    # epsilon_vals = [0.07, 0.23, 0.52, 0.74, 1]
    epsilon_vals = [1, 10]


    dataset_name = 'folktables_2018_multitask_NY'
    root_path = '../../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)
    df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
    df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')

    print(f'train size: {df_train.shape}')
    print(f'test size:  {df_test.shape}')
    domain = Domain.fromdict(config)
    data = Dataset(df_train, domain)

    targets = ['PINCP',  'PUBCOV', 'ESR']
    #############
    ## ML Function
    ml_eval_fn = get_evaluate_ml(df_test, config, targets=targets, models=['LogisticRegression'])

    orig_train_results = ml_eval_fn(df_train, 0)
    print(f'Original train data ML results:')
    print(orig_train_results)

    # Create statistics and evaluate
    key = jax.random.PRNGKey(0)
    module0 = Marginals.get_all_kway_combinations(data.domain, k=2, bins=[2, 4, 8, 16, 32])
    stat_module = ChainedStatistics([module0])
    stat_module.fit(data)

    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module._get_workload_fn()

    algo = GeneticSD(num_generations=40000, print_progress=False, stop_early=True, strategy=GeneticStrategy(domain=data.domain, elite_size=2, data_size=1000))
    # Choose algorithm parameters

    delta = 1.0 / len(data) ** 2
    # Generate differentially private synthetic data with ADAPTIVE mechanism
    # for eps in [1]:
    for eps in epsilon_vals:
        for seed in [0, 1, 2]:
        # for seed in [0]:
            sync_dir = f'sync_data/{dataset_name}/PrivGA/Ranges/oneshot/{eps:.2f}/'
            os.makedirs(sync_dir, exist_ok=True)

            key = jax.random.PRNGKey(seed)
            t0 = timer()

            def debug(i, sync_data: Dataset):
                print(f'results epoch {i}:')
                results = ml_eval_fn(sync_data.df, seed)
                results = results[results['Eval Data'] == 'Test']
                print(results)

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

