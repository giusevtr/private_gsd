import itertools
import jax
from models import Generator, PrivGA, SimpleGAforSyncData
from stats import Marginals
from toy_datasets.sparse import get_sparse_dataset
import matplotlib.pyplot as plt
import time


PRINT_PROGRESS = True
ROUNDS = 1
EPSILON = [1.00]
SEEDS = [0]




if __name__ == "__main__":

    # rng = np.random.default_rng()
    # data_np = np.column_stack((rng.uniform(low=0.20, high=0.21, size=(10000, )),
    #                            rng.uniform(low=0.30, high=0.80, size=(10000, ))))
    BINS = 30
    data = get_sparse_dataset(DATA_SIZE=10000)
    # plot_2d_data(data.to_numpy())

    stats_module = Marginals.get_all_kway_combinations(data.domain, 3, bins=BINS)

    stats_module.fit(data)


    data_size = 1000
    strategy = SimpleGAforSyncData(
            domain=data.domain,
            data_size=data_size,
            population_size=100,
            elite_size=10,
            muta_rate=1,
            mate_rate=100
        )

    ########
    # PrivGA
    ########
    priv_ga = PrivGA(
                    num_generations=10000,
                    stop_loss_time_window=50,
                    print_progress=True,
                    strategy=strategy
                     )



    # plot_2d_data(data_np)

    def plot_sparse(data_array, alpha=0.9, title='', save_path=None):
        plt.figure(figsize=(5, 5))
        plt.title(title)
        plt.scatter(data_array[:, 0], data_array[:, 1], c=data_array[:, 2], alpha=alpha, s=0.1)
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
            plot_sparse(X, title=f'RAP, eps={eps:.2f}, epoch={t:03}')


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
        print(f'{str(priv_ga)}: max error = {erros.max():.5f}, time={time.time()-stime}')

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