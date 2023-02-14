import itertools
import jax
from models import Generator, RelaxedProjectionPP
from stats import Halfspace
from dev.toy_datasets.sparse import get_sparse_dataset
import time
from plot import plot_sparse


PRINT_PROGRESS = True
ROUNDS = 30
EPSILON = [1]
# EPSILON = [1]
SEEDS = [0, 1, 2]


if __name__ == "__main__":

    data = get_sparse_dataset(DATA_SIZE=10000)
    key_hs = jax.random.PRNGKey(0)
    stats_module, kway_combinations = Halfspace.get_kway_random_halfspaces(data.domain, k=0, rng=key_hs,
                                                                           random_hs=1000)
    stats_module.fit(data)
    print(f'workloads = ', len(stats_module.true_stats))
    print(f'stats size = ', stats_module.get_true_stats().shape)
    data_size = 500
    rappp = RelaxedProjectionPP(
        domain=data.domain,
        data_size=data_size,
        learning_rate=(0.0001,),
        print_progress=False,
        )

    RESULTS = []
    for eps, seed in itertools.product(EPSILON, SEEDS):
        priv_ga: Generator

        def debug_fn(t, tempdata):
            plot_sparse(tempdata.to_numpy(), title=f'epoch={t}, RAP++, Halfspace, eps={eps:.2f}',
                    alpha=0.9, s=0.8)

        print(f'Starting {rappp}:')
        stime = time.time()
        key = jax.random.PRNGKey(seed)

        sync_data = rappp.fit_dp_adaptive(key, stat_module=stats_module,  epsilon=eps, delta=1e-6,
                                            rounds=ROUNDS, print_progress=True, debug_fn=debug_fn)
        plot_sparse(sync_data.to_numpy(), title=f'PrivGA, Halfspaces, eps={eps:.2f}', alpha=0.9, s=0.8)

        errors = jax.numpy.abs(stats_module.get_true_stats() - stats_module.get_stats_jit(sync_data))
        ave_error = jax.numpy.linalg.norm(errors, ord=1)
        print(f'{str(rappp)}: max error = {errors.max():.4f}, ave_error={ave_error:.6f}, time={time.time()-stime:.4f}')