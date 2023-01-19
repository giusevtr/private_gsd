import itertools
import jax
from models import Generator, RelaxedProjection
from stats import Marginals
from toy_datasets.sparsecat import get_sparsecat
import matplotlib.pyplot as plt
import time


PRINT_PROGRESS = True
ROUNDS = 2
EPSILON = [2]
# EPSILON = [1]
SEEDS = [0]


if __name__ == "__main__":

    data = get_sparsecat(DATA_SIZE=2000)

    stats_module, kway_combinations = Marginals.get_all_kway_combinations(data.domain, k=2)
    stats_module.fit(data)

    plt.hist(data.to_numpy())
    plt.show()

    print(f'workloads = ', len(stats_module.true_stats))
    data_size = 1000
    rap = RelaxedProjection(
            domain=data.domain,
            data_size=data_size,
            learning_rate=0.3,
                print_progress=True
        )


    RESULTS = []
    for eps, seed in itertools.product(EPSILON, SEEDS):
        priv_ga: Generator

        print(f'Starting {rap}:')
        stime = time.time()

        key = jax.random.PRNGKey(seed)
        sync_data = rap.fit_dp_adaptive(key, stat_module=stats_module, epsilon=eps, delta=1e-6, rounds=ROUNDS,
                                            start_sync=True,
                                            print_progress=True)
        erros = stats_module.get_sync_data_errors(sync_data.to_numpy())

        stats = stats_module.get_stats_jit(sync_data)
        ave_error = jax.numpy.linalg.norm(stats_module.get_true_stats() - stats, ord=1)
        print(f'{str(rap)}: max error = {erros.max():.4f}, ave_error={ave_error:.6f}, time={time.time()-stime:.4f}')

