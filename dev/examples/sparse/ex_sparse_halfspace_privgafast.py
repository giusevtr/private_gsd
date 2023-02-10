import itertools
import jax
from models import Generator, PrivGAfast, SimpleGAforSyncDataFast
from stats.halfspaces_v3 import Halfspace3
from dev.toy_datasets.sparse import get_sparse_dataset
import time
from plot import plot_sparse


PRINT_PROGRESS = True
ROUNDS = 30
EPSILON = [1]
# EPSILON = [1]
SEEDS = [0]


if __name__ == "__main__":

    data = get_sparse_dataset(DATA_SIZE=10000)
    key_hs = jax.random.PRNGKey(0)
    stats_module, kway_combinations = Halfspace3.get_kway_random_halfspaces(data.domain, k=1, rng=key_hs,
                                                                           random_hs=1000)
    stats_module.fit(data)
    print(f'workloads = ', len(stats_module.true_stats))
    data_size = 500
    strategy = SimpleGAforSyncDataFast(
            domain=data.domain,
            data_size=data_size,
            population_size=100,
            elite_size=5,
            muta_rate=1,
            mate_rate=1,
        )
    priv_ga = PrivGAfast(
                    num_generations=10000,
                    strategy=strategy,
                    print_progress=False,
                     )

    RESULTS = []
    for eps, seed in itertools.product(EPSILON, SEEDS):
        priv_ga: Generator

        def debug_fn(t, tempdata):
            plot_sparse(tempdata.to_numpy(), title=f'epoch={t}, PrivGA, Prefix, eps={eps:.2f}',
                    alpha=0.9, s=0.8)

        print(f'Starting {priv_ga}:')
        stime = time.time()
        key = jax.random.PRNGKey(seed)

        sync_data = priv_ga.fit_dp_adaptive(key, stat_module=stats_module,  epsilon=eps, delta=1e-6,
                                            rounds=ROUNDS, print_progress=True, debug_fn=debug_fn)
        plot_sparse(sync_data.to_numpy(), title=f'PrivGA, Halfspaces, eps={eps:.2f}', alpha=0.9, s=0.8)

        errors = jax.numpy.abs(stats_module.get_true_stats() - stats_module.get_stats_jit(sync_data))
        ave_error = jax.numpy.linalg.norm(errors, ord=1)
        print(f'{str(priv_ga)}: max error = {errors.max():.4f}, ave_error={ave_error:.6f}, time={time.time()-stime:.4f}')