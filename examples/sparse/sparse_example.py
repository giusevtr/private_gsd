import itertools
import sys

import pandas as pd
import jax
from models_v3 import Generator, PrivGA
from stats_v3 import Marginals
from toy_datasets.circles import get_circles_dataset
from visualize.plot_low_dim_data import plot_2d_data
from toy_datasets.sparse import get_sparse_dataset
import seaborn as sns
import matplotlib.pyplot as plt
import time
PRINT_PROGRESS = True
ROUNDS = 1
EPSILON = [0.07]
SEEDS = [0]


ALGO = [
    #  PrivGAIP(
    #     popsize=100,
    #     top_k=20,
    #     num_generations=5000,
    #     stop_loss_time_window=20,
    #     print_progress=PRINT_PROGRESS,
    #     start_mutations=2,
    #     data_size=200,
    # ),
    PrivGA(
        popsize=100,
        top_k=5,
        num_generations=5000,
        stop_loss_time_window=20,
        print_progress=PRINT_PROGRESS,
        start_mutations=32,
        data_size=200,
    ),
    # RelaxedProjectionPP(data_size=100,
    #                     iterations=5000,
    #                     learning_rate=(0.001, 0.005, 0.01),
    #                     print_progress=PRINT_PROGRESS,
    #                     early_stop_percent=0.001)
]

if __name__ == "__main__":

    # rng = np.random.default_rng()
    # data_np = np.column_stack((rng.uniform(low=0.20, high=0.21, size=(10000, )),
    #                            rng.uniform(low=0.30, high=0.80, size=(10000, ))))

    data = get_sparse_dataset(DATA_SIZE=10000)
    # plot_2d_data(data.to_numpy())

    stats_module = Marginals.get_all_kway_combinations(data.domain, 2, bins=30)

    stats_module.fit(data)

    def plot_circles(X):
        plot_2d_data(X)

    # plot_2d_data(data_np)

    def plot_sparse(data_array, alpha=1.0, title='', save_path=None):
        plt.figure(figsize=(5, 5))
        plt.title(title)
        plt.scatter(data_array[:, 0], data_array[:, 1], c=data_array[:, 2], alpha=alpha, s=0.7)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
        plt.close()

    RESULTS = []
    for algo, eps, seed in itertools.product(ALGO, EPSILON, SEEDS):
        algo: Generator

        def debug_fn(t, X):
            plot_2d_data(X, title=f'{str(algo)}, eps={eps:.2f}, epoch={t:03}')
            pass

        ##############
        ## Non-Regularized
        ##############
        print(f'Starting {str(algo)}:')
        stime = time.time()

        key = jax.random.PRNGKey(seed)
        sync_data = algo.fit_dp_adaptive(key, stat_module=stats_module, rounds=ROUNDS, epsilon=eps, delta=1e-6,
                                            tolerance=0.01,
                                         print_progress=True, debug_fn=debug_fn)
        erros = stats_module.get_sync_data_errors(sync_data.to_numpy())
        print(f'{str(algo)}: max error = {erros.max():.5f}, time={time.time()-stime}')

        df = algo.ADA_DATA
        df['algo'] = str(algo)
        df['eps'] = eps
        df['seed'] = seed
        RESULTS.append(df)


    types = {'average error': float, 'max error': float, 'algo': str}
    # df_all = pd.concat(RESULTS, ignore_index=True)
    # df_all = df_all.astype(types)
    # df_all.to_csv('sparse_results.csv', index=False)
    # # df_melt =
    # df_melt = pd.melt(df_all, var_name='error type', value_name='error', id_vars=['epoch', 'algo', 'eps', 'seed'])
    #
    # sns.relplot(data=df_melt, x='epoch', y='error', hue='algo', col='error type', row='eps', kind='line',
    #             facet_kws={'sharey': False, 'sharex': True})
    # plt.show()