import itertools
import jax
from models import Generator, PrivGAfast, SimpleGAforSyncDataFast
from stats import Marginals
from toy_datasets.sparsecat import get_sparsecat
import matplotlib.pyplot as plt
import time


PRINT_PROGRESS = True
ROUNDS = 2
EPSILON = [0.5]
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
    strategy = SimpleGAforSyncDataFast(
            domain=data.domain,
            data_size=data_size,
            population_size=200,
            elite_size=20,
            muta_rate=1,
            mate_rate=1,
            debugging=False
        )
    priv_ga = PrivGAfast(
                    num_generations=100000,
                    strategy=strategy,
                    print_progress=True,
                     )


    RESULTS = []
    for eps, seed in itertools.product(EPSILON, SEEDS):
        priv_ga: Generator

        print(f'Starting {priv_ga}:')
        stime = time.time()

        key = jax.random.PRNGKey(seed)
        sync_data = priv_ga.fit_dp_adaptive(key, stat_module=stats_module, epsilon=eps, delta=1e-6, rounds=ROUNDS,
                                            print_progress=True)
        erros = stats_module.get_sync_data_errors(sync_data.to_numpy())

        stats = stats_module.get_stats_jit(sync_data)
        ave_error = jax.numpy.linalg.norm(stats_module.get_true_stats() - stats, ord=1)
        print(f'{str(priv_ga)}: max error = {erros.max():.4f}, ave_error={ave_error:.6f}, time={time.time()-stime:.4f}')

