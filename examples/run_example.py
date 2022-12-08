"""
End-to-end private example
"""
import itertools
import os
import pandas as pd
from utils.cdp2adp import cdp_rho
import numpy as np
import jax.numpy as jnp
import jax
from models import Generator
from typing import Tuple, Any, Callable
import time
from utils import Dataset, Domain
import matplotlib.pyplot as plt
import seaborn as sns


def save_csv_fn(df: pd.DataFrame, dataname, algorithm, stats, epsilon, seed, runtime=0):

    path = [dataname, algorithm, stats, epsilon, seed]
    result_dir = ''
    for p in path:
        result_dir += p
        os.makedirs(result_dir, exist_ok=True)
    result_path = f'{result_dir}/sync.csv'
    # Save runtime.
    df.to_csv(result_path, index=False)

def run_experiment(
        data: Dataset,
        data_name: str,
        generators: list,
        stat_modules: list,
        epsilon_list: list,
        seed_list: list,
        data_size=30,
        save_results=False,
        plot_results=False,
        callback_fn=None
                   ):

    RESULTS = []

    for get_gen, stat_module, seed in itertools.product(generators, stat_modules, seed_list):
        generator = get_gen(data.domain, stat_module, data_size=data_size, seed=seed)
        for epsilon in epsilon_list:
            X_sync, error, max_error, elapsed_time = generate_private_SD(data, generator,  epsilon, seed)
            if callback_fn is not None:
                callback_fn(X_sync)
                RESULTS.append(
                    [data_name, str(generator), str(stat_module), epsilon, seed, float(error), float(max_error),elapsed_time])
        results_df = pd.DataFrame(RESULTS,
                                  columns=['data', 'generator', 'stats', 'epsilon', 'seed', 'l1 error', 'max error', 'time'])
    #         RESULTS.append([data_name, str(generator), str(stat_module), epsilon, seed, float(error), float(max_error),generator.top_k,generator.crossover, generator.mutations, elapsed_time])
    # results_df = pd.DataFrame(RESULTS, columns=['data', 'generator', 'stats', 'epsilon', 'seed', 'l1 error', 'max error','top_k','crossover','mutations', 'time'])

    # results_df.to_csv(f'results/results_{data_name}_{str(stat_modules[0])}_parameters.csv', index_label=False)
    if plot_results:
        plt.title(f'data={data_name}')
        sns.relplot(data=results_df, x='epsilon', y='l1 error', hue='generator', row='data', col='stats', kind='line')
        if os.path.exists('results/'):
            plt.savefig(f'results/{data_name}_{str(stat_modules[0])}_l1.png')
        plt.show()
        sns.relplot(data=results_df, x='epsilon', y='max error', hue='generator', row='data', col='stats', kind='line')
        if os.path.exists('results/'):
            plt.savefig(f'results/{data_name}_{str(stat_modules[0])}_max.png')
        plt.show()
    return results_df

def generate_private_SD(
        data: Dataset,
        generator: Generator,
        # stat_module,
        epsilon,
        seed=0,
    ):

    X = data.to_numpy()
    stime = time.time()
    n, dim = X.shape

    init_X = None

    n = X.shape[0]
    delta = 1 / (n ** 2)
    rho = cdp_rho(epsilon, delta)
    rng = np.random.default_rng(seed)

    # Get statistics and noisy statistics
    stat_module = generator.stat_module
    stats_fn = jax.jit(stat_module.get_stats_fn())
    true_stats = stats_fn(X)

    num_queries = true_stats.shape[0]
    sensitivity = stat_module.get_sensitivity() / n
    print(f'Running {str(generator)}:')
    print(f'Stats is {str(stat_module)}')
    print(f'dimension = {dim}, num queries = {num_queries}, sensitivity={sensitivity:.4f}')

    # sigma_gaussian = float(np.sqrt(num_queries / (2 * (n ** 2) * rho)))
    sigma_gaussian = float(np.sqrt(sensitivity**2 / (2 * rho)))

    true_stats_noise = true_stats + rng.normal(size=true_stats.shape) * sigma_gaussian
    gaussian_error = jnp.abs(true_stats - true_stats_noise)
    print(f'epsilon={epsilon}, '
          f'Gaussian error: L1 {jnp.linalg.norm(gaussian_error, ord=1):.5f},'
          f'Gaussian error: L2 {jnp.linalg.norm(gaussian_error, ord=2):.5f},'
          f' Max = {gaussian_error.max():.5f}')


    dataset: Dataset
    dataset = generator.fit(true_stats_noise,  init_X=init_X)
    X_sync = dataset.to_numpy()

    elapsed_time = time.time() - stime

    # Use large statistic set to evaluate the final synthetic data.
    # evaluate_stats_fn = stat_module.get_stats_fn(num_cols=dim, num_rand_queries=10000, seed=0)
    evaluate_true_stats = stats_fn(X)
    sync_stats = stats_fn(X_sync)
    errors = jnp.abs(evaluate_true_stats - sync_stats)
    error = jnp.linalg.norm(errors, ord=1)
    error_l2 = jnp.linalg.norm(errors, ord=2)
    max_error = errors.max()
    print(f'Final L1 error = {error:.5f}, L2 error = {error_l2:.5f},  max error ={max_error:.5f}\n')


    return X_sync, error, max_error, elapsed_time