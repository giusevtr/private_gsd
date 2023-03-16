import itertools
import jax
# from models import Generator, RelaxedProjectionPP
# from models import RelaxedProjectionPPneurips as RelaxedProjectionPP
from models import RelaxedProjectionPP as RelaxedProjectionPP
from stats import HalfspaceDiff, PrefixDiff, ChainedStatistics, MarginalsDiff, Prefix
from dev.toy_datasets.sparse import get_sparse_dataset
import time
from plot import plot_sparse
import jax.numpy as jnp
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

PRINT_PROGRESS = True
ROUNDS = 50
SAMPLES = 10
EPSILON = [10]
# EPSILON = [1]
SEEDS = [0]


if __name__ == "__main__":

    data = get_sparse_dataset(DATA_SIZE=10000)
    key_hs = jax.random.PRNGKey(0)
    # module = HalfspaceDiff.get_kway_random_halfspaces(data.domain, k=1, rng=key_hs,
    #                                                                        random_hs=1000)

    module0 = MarginalsDiff.get_all_kway_categorical_combinations(data.domain, k=1)
    module = PrefixDiff.get_kway_prefixes(data.domain, k_cat=1, k_num=2, rng=key_hs, random_prefixes=1000)
    module_pre = Prefix.get_kway_prefixes(data.domain, k_cat=1, k_num=2, rng=key_hs, random_prefixes=1000)

    stats_module = ChainedStatistics([
        module0,
        module,
                                     # module1
                                     ])
    stats_module.fit(data)

    true_stats = stats_module.get_all_true_statistics()
    stat_fn = stats_module.get_dataset_statistics_fn()

    data_size = 500
    rappp = RelaxedProjectionPP(
        domain=data.domain,
        data_size=data_size,
        iterations=5000,
        learning_rate=(0.005, 0.001),
        print_progress=True,
        )

    RESULTS = []
    for eps, seed in itertools.product(EPSILON, SEEDS):

        def debug_fn(t, tempdata):
            plot_sparse(tempdata.to_numpy(), title=f'epoch={t}, RAP++, Prefix, eps={eps:.2f}',
                    alpha=0.9, s=0.8)

        print(f'Starting {rappp}:')
        stime = time.time()
        key = jax.random.PRNGKey(seed)

        # sync_data = rappp.fit_dp_adaptive(key, stat_module=stats_module,  epsilon=eps, delta=1e-6,
        #                                     rounds=ROUNDS, num_sample=SAMPLES, print_progress=True, debug_fn=debug_fn)
        sync_data = rappp.fit_dp_hybrid(key, stat_module=stats_module,  epsilon=eps, delta=1e-6,
                                            rounds=ROUNDS, num_sample=SAMPLES, print_progress=True, debug_fn=debug_fn)

        plot_sparse(sync_data.to_numpy(), title=f'RAP++, Prefix, eps={eps:.2f}', alpha=0.9, s=0.8)

        errors = jax.numpy.abs(true_stats - stat_fn(sync_data))
        ave_error = jax.numpy.linalg.norm(errors, ord=1)
        print(f'{str(rappp)}: max error = {errors.max():.4f}, ave_error={ave_error:.6f}, time={time.time()-stime:.4f}')