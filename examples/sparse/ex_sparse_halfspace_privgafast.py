import itertools
import jax
from models import Generator, PrivGAfast, SimpleGAforSyncDataFast
from stats import Halfspace
from toy_datasets.sparse import get_sparse_dataset
from utils import Dataset
import matplotlib.pyplot as plt
import time
from plot import plot_sparse


PRINT_PROGRESS = True
ROUNDS = 15
EPSILON = [0.1]
# EPSILON = [1]
SEEDS = [0]


if __name__ == "__main__":

    data = get_sparse_dataset(DATA_SIZE=10000)
    key_hs = jax.random.PRNGKey(0)
    stats_module, kway_combinations = Halfspace.get_kway_random_halfspaces(data.domain, k=3, rng=key_hs,
                                                                           hs_workloads=300,
                                                                           random_hs=1)
    stats_module.fit(data)


    def regu_get_fn():
        regu_fn = lambda X: stats_module.get_stats_jax(X)
        return regu_fn

    print(f'workloads = ', len(stats_module.true_stats))
    data_size = 200
    strategy = SimpleGAforSyncDataFast(
            domain=data.domain,
            data_size=data_size,
            population_size=100,
            elite_size=5,
            muta_rate=1,
            mate_rate=1,
                debugging=False
        )
    priv_ga = PrivGAfast(
                    num_generations=10000,
                    strategy=strategy,
                    print_progress=False,
                    regu_get_fn=regu_get_fn,
                     )



    RESULTS = []
    for eps, seed in itertools.product(EPSILON, SEEDS):
        priv_ga: Generator
        print(f'Starting {priv_ga}:')
        stime = time.time()

        def debug_fn(t, tempdata):
            plot_sparse(tempdata.to_numpy(), title=f'epoch={t}, PrivGA, Prefix, eps={eps:.2f}',
                    alpha=0.9, s=0.8)


        key = jax.random.PRNGKey(seed)


        sync_data = priv_ga.fit_dp_adaptive(key, stat_module=stats_module,  epsilon=eps, delta=1e-6,
                                            rounds=ROUNDS, print_progress=True, debug_fn=debug_fn)
        plot_sparse(sync_data.to_numpy(), title=f'PrivGA, Halfspaces, eps={eps:.2f}', alpha=0.9, s=0.8)

        errors = jax.numpy.abs(stats_module.get_true_stats() - stats_module.get_stats_jit(sync_data))
        ave_error = jax.numpy.linalg.norm(errors, ord=1)
        print(f'{str(priv_ga)}: max error = {errors.max():.4f}, ave_error={ave_error:.6f}, time={time.time()-stime:.4f}')



    # types = {'average error': float, 'max error': float, 'algo': str}
    # df_all = pd.concat(RESULTS, ignore_index=True)
    # df_all = df_all.astype(types)
    # df_all.to_csv('sparse_results.csv', index=False)
    # # df_melt =
    # df_melt = pd.melt(df_all, var_name='error type', value_name='error', id_vars=['epoch', 'algo', 'eps', 'seed'])
    #
    # sns.relplot(data=df_melt, x='epoch', y='error', hue='algo', col='error type', row='eps', kind='line',
    #             facet_kws={'sharey': False, 'sharex': True})
    # plt.show()