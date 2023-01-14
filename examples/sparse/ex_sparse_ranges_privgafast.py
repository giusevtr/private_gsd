import itertools
import jax
from models import Generator, PrivGAfast, SimpleGAforSyncDataFast
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

    data = get_sparse_dataset(DATA_SIZE=10000)
    bins = [2, 4, 8, 16, 32, 64]
    stats_module, kway_combinations = Marginals.get_all_kway_mixed_combinations(data.domain, k_disc=1, k_real=2,
                                                                                bins=bins)
    stats_module.fit(data)
    print(f'workloads = ', len(stats_module.true_stats))
    data_size = 200
    strategy = SimpleGAforSyncDataFast(
            domain=data.domain,
            data_size=data_size,
            population_size=100,
            elite_size=5,
            muta_rate=1,
            mate_rate=5,
                debugging=False
        )
    priv_ga = PrivGAfast(
                    num_generations=10000,
                    strategy=strategy,
                    print_progress=True,
                     )
    priv_ga.early_stop_elapsed_time = 1


    RESULTS = []
    for eps, seed in itertools.product(EPSILON, SEEDS):
        priv_ga: Generator

        print(f'Starting {priv_ga}:')
        stime = time.time()

        key = jax.random.PRNGKey(seed)
        sync_data = priv_ga.fit_dp(key, stat_module=stats_module, epsilon=eps, delta=1e-6)
        erros = stats_module.get_sync_data_errors(sync_data.to_numpy())
        plot_sparse(sync_data.to_numpy(), title=f'PrivGA, Ranges with {max(bins)} bins, eps={eps:.2f}', alpha=0.9, s=0.8)

        stats = stats_module.get_stats_jit(sync_data)
        ave_error = jax.numpy.linalg.norm(stats_module.get_true_stats() - stats, ord=1)
        print(f'{str(priv_ga)}: max error = {erros.max():.4f}, ave_error={ave_error:.6f}, time={time.time()-stime:.4f}')

