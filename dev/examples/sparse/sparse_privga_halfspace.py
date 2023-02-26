import itertools
import jax
from models import Generator, PrivGA, SimpleGAforSyncData
from stats import Halfspace, ChainedStatistics
from dev.toy_datasets.sparse import get_sparse_dataset
import time
from plot import plot_sparse


PRINT_PROGRESS = True
ROUNDS = 10
SAMPLES = 3
EPSILON = [10]
# EPSILON = [1]
SEEDS = [0]


if __name__ == "__main__":

    data = get_sparse_dataset(DATA_SIZE=10000)
    key_hs = jax.random.PRNGKey(0)
    module = Halfspace.get_kway_random_halfspaces(data.domain, k=1, rng=key_hs,
                                                                           random_hs=1000)

    stats_module = ChainedStatistics([module,
                                     # module1
                                     ])
    stats_module.fit(data)

    true_stats = stats_module.get_all_true_statistics()
    stat_fn = stats_module.get_dataset_statistics_fn()

    data_size = 500

    algo = PrivGA(num_generations=40000, print_progress=False, stop_early=True, strategy=SimpleGAforSyncData(domain=data.domain, elite_size=5, data_size=2000))

    RESULTS = []
    for eps, seed in itertools.product(EPSILON, SEEDS):

        def debug_fn(t, tempdata):
            plot_sparse(tempdata.to_numpy(), title=f'epoch={t}, RAP++, Halfspace, eps={eps:.2f}',
                    alpha=0.9, s=0.8)

        print(f'Starting {algo}:')
        stime = time.time()
        key = jax.random.PRNGKey(seed)

        sync_data = algo.fit_dp_adaptive(key, stat_module=stats_module,  epsilon=eps, delta=1e-6,
                                            rounds=ROUNDS, num_sample=SAMPLES, print_progress=True, debug_fn=debug_fn)
        plot_sparse(sync_data.to_numpy(), title=f'PrivGA, Halfspaces, eps={eps:.2f}', alpha=0.9, s=0.8)

        errors = jax.numpy.abs(true_stats - stat_fn(sync_data))
        ave_error = jax.numpy.linalg.norm(errors, ord=1)
        print(f'{str(algo)}: max error = {errors.max():.4f}, ave_error={ave_error:.6f}, time={time.time()-stime:.4f}')