import jax
from models import GeneticSD, GeneticStrategy
from stats import Marginals, ChainedStatistics
from dev.toy_datasets.sparsecat import get_sparsecat
import time
import jax.numpy as jnp


PRINT_PROGRESS = True
ROUNDS = 2
EPSILON = [1.5]
# EPSILON = [1]
SEEDS = [0]


def run(data, algo, marginal_module, eps=0.5, seed=0):
    print(f'Running {algo} with {marginal_module}')

    stat_module = ChainedStatistics([
                                         marginal_module,
                                         ])
    stat_module.fit(data)
    true_stats = stat_module.get_all_true_statistics()
    stat_fn = marginal_module._get_dataset_statistics_fn()


    stime = time.time()

    key = jax.random.PRNGKey(seed)
    # sync_data = algo.fit_dp(key, stat_module=stat_module, epsilon=eps, delta=1/len(data.df)**2)
    sync_data = algo.fit_dp_adaptive(key, stat_module=stat_module, epsilon=eps, delta=1/len(data.df)**2, rounds=3, print_progress=True)

    errors = jnp.abs(true_stats - stat_fn(sync_data))
    print(f'{algo}: eps={eps:.2f}, seed={seed}'
          f'\t max error = {errors.max():.5f}'
          f'\t avg error = {jnp.linalg.norm(errors, ord=2):.5f}')

    sel_true_stats = stat_module.get_selected_statistics_without_noise()
    noisy_stats = stat_module.get_selected_noised_statistics()
    gau_errors = jnp.abs(sel_true_stats - noisy_stats)
    print(f'\t Gau max error = {gau_errors.max():.5f}')
    print(f'\t Gau ave error = {jnp.linalg.norm(gau_errors, ord=2):.5f}')


if __name__ == "__main__":

    data = get_sparsecat(DATA_SIZE=2000, CAT_SIZE=10)


    # stats = MarginalsDiff.get_all_kway_combinations(data.domain, k=2)
    # algo = RelaxedProjection(data.domain, 1000, 2000, learning_rate=0.01, print_progress=True)
    # run(data, algo, stats, seed=1)



    marginal_module = Marginals.get_all_kway_combinations(data.domain, k=2)
    priv_ga = GeneticSD(num_generations=100000, print_progress=False, strategy=GeneticStrategy(
                        domain=data.domain, data_size=2000,
                        population_size=200))
    run(data, priv_ga, marginal_module, eps=5, seed=1)


