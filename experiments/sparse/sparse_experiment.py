import itertools

import folktables
import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSEmployment
from utils import Dataset, Domain, DataTransformer
import jax.numpy as jnp
from models_v2 import Generator, PrivGA, RelaxedProjectionPP
from stats_v2 import TwoWayPrefix
from utils.utils_data import get_data
import sys, os
import jax
from visualize.plot_low_dim_data import plot_2d_data
from toy_datasets.sparse import get_sparse_dataset
import matplotlib.pyplot as plt
import seaborn as sns

PRINT_PROGRESS = True
ALGORITHMS = [
    PrivGA(popsize=100,
            top_k=50,
            num_generations=2000,
            stop_loss_time_window=30,
            print_progress=PRINT_PROGRESS,
            start_mutations=32,
            data_size=100),
    RelaxedProjectionPP(data_size=100,
                        iterations=5000,
                        learning_rate=(0.001, 0.005, 0.01),
                        print_progress=PRINT_PROGRESS,
                        early_stop_percent=0.001)
]

def run_experiments(epsilon=(0.07, 0.15, 0.25), query_size=5000):
    data = get_sparse_dataset(DATA_SIZE=10000, seed=0)
    data_name = 'sparse_2d'
    plot_2d_data(data.to_numpy())
    RESULTS = []

    stats_module = TwoWayPrefix.get_stat_module(data.domain, num_rand_queries=query_size)
    stats_module.fit(data)

    for T, eps, seed in itertools.product([ 50, 80], list(epsilon), [0, 1, 2]):


        for algorithm in ALGORITHMS:
            algorithm: Generator
            key = jax.random.PRNGKey(seed)
            sync_data_2 = algorithm.fit_dp_adaptive(key, stat_module=stats_module,
                                                  rounds=T, epsilon=eps, delta=1e-6, print_progress=PRINT_PROGRESS)

            sync_error = stats_module.get_sync_data_errors(sync_data_2.to_numpy())
            average_error = jnp.linalg.norm(sync_error, ord=1) / sync_error.shape[0]
            max_error = sync_error.max()

            print(f'{str(algorithm)}. Average error {average_error:.5f} max error = {max_error:.4f}')
            this_result = [data_name, str(algorithm),  T, eps, seed, average_error, max_error]
            RESULTS.append(this_result)


            algo_name = str(algorithm)
            save_path = '../sync_datasets'
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, data_name)
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, algo_name)
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, f'{T:03}')
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, f'{eps:.2f}')
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, f'sync_data_{seed}.csv')
            data_df: pd.DataFrame = sync_data_2.df
            print(f'Saving {save_path}')
            data_df.to_csv(save_path)


    results_df = pd.DataFrame(RESULTS, columns=['data name', 'algo', 'T', 'epsilon', 'seed', 'average error', 'max error'])
    return results_df


if __name__ == "__main__":
    # df = folktables.
    df: pd.DataFrame = run_experiments()

    types = {'average error': float, 'max error': float, 'algo': str}
    df = df.astype(types)
    df.to_csv('results.csv')

    df_melt = pd.melt(df, var_name='error type', value_name='error', id_vars=['data name', 'T', 'algo', 'epsilon', 'seed'])
    # sns.relplot(data = df, x='epsilon', )
    sns.relplot(data=df_melt, x='epsilon', y='error', hue='algo', col='error type', row='T', kind='line',
                facet_kws={'sharey': False, 'sharex': True})

    plt.savefig('result.png')


