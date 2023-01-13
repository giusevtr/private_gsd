import itertools
import jax
from models import Generator, PrivGA, SimpleGAforSyncData
from stats import Marginals
from toy_datasets.sparse import get_sparse_dataset
import matplotlib.pyplot as plt
import time
from plot import plot_sparse


PRINT_PROGRESS = True
ROUNDS = 1
EPSILON = [0.1]
# EPSILON = [1]
SEEDS = [0]




if __name__ == "__main__":

    # rng = np.random.default_rng()
    # data_np = np.column_stack((rng.uniform(low=0.20, high=0.21, size=(10000, )),
    #                            rng.uniform(low=0.30, high=0.80, size=(10000, ))))
    # BINS = 32
    data = get_sparse_dataset(DATA_SIZE=10000)
    # plot_2d_data(data.to_numpy())
    plot_sparse(data.to_numpy(), title='Original sparse')
    bins = [2, 4, 8, 16, 32, 64]

    stats_module, kway_combinations = Marginals.get_all_kway_mixed_combinations(data.domain, k_disc=1, k_real=2,
                                                                                bins=bins)
    stats_module.fit(data)
    print(f'workloads = ', len(stats_module.true_stats))
    data_size = 2000
    strategy = SimpleGAforSyncData(
            domain=data.domain,
            data_size=data_size,
            population_size=100,
            elite_size=10,
            muta_rate=1,
            mate_rate=10
        )
    priv_ga = PrivGA(
                    num_generations=10000,
                    stop_loss_time_window=50,
                    print_progress=True,
                    strategy=strategy,
                     )



    # plot_2d_data(data_np)

    def plot_sparse(data_array, alpha=0.9, title='', save_path=None):
        plt.figure(figsize=(5, 5))
        plt.title(title)
        plt.scatter(data_array[:, 0], data_array[:, 1], c=data_array[:, 2], alpha=alpha, s=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
        plt.close()

    RESULTS = []
    for eps, seed in itertools.product(EPSILON, SEEDS):
        priv_ga: Generator


        def debug_fn(t, sync_dataset):
            X = sync_dataset.to_numpy()
            plot_sparse(X, title=f'PrivGA, eps={eps:.2f}, epoch={t:03}')


        ##############
        ## Non-Regularized
        ##############
        print(f'Starting PrivGA:')
        stime = time.time()

        key = jax.random.PRNGKey(seed)
        sync_data = priv_ga.fit_dp_adaptive(key, stat_module=stats_module, rounds=ROUNDS, epsilon=eps, delta=1e-6,
                                            tolerance=0.0,
                                         print_progress=True, debug_fn=debug_fn)
        erros = stats_module.get_sync_data_errors(sync_data.to_numpy())

        stats = stats_module.get_stats_jit(sync_data)
        ave_error = jax.numpy.linalg.norm(stats_module.get_true_statistics() - stats, ord=1)
        print(f'{str(priv_ga)}: max error = {erros.max():.4f}, ave_error={ave_error:.6f}, time={time.time()-stime:.4f}')

        df = priv_ga.ADA_DATA
        df['algo'] = str(priv_ga)
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