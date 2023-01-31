import jax.random
from models import PrivGA, SimpleGAforSyncData
from stats import Marginals
from utils.utils_data import get_data
from utils import timer
from toy_datasets.classification import get_classification
import jax.numpy as jnp
from utils.plot_low_dim_data import plot_2d_data



if __name__ == "__main__":

    # Get Data
    ROUNDS = 50
    # data_name = f'folktables_2018_CA'
    data_name = 'folktables_2018_coverage_CA'
    data = get_data(f'{data_name}-mixed-train', domain_name=f'domain/{data_name}-cat',  root_path='../data_files/folktables_datasets')
    # data = get_classification(DATA_SIZE=1000)

    SYNC_DATA_SIZE = 2000
    # Create statistics and evaluate
    marginal_module, _ = Marginals.get_all_kway_combinations(data.domain, k=2, bins=[2, 4, 8, 16, 32])
    marginal_module.fit(data)

    true_stats = marginal_module.get_true_statistics()
    stat_fn = marginal_module._get_workload_fn()

    # Choose algorithm parameters
    priv_ga = PrivGA(
                    num_generations=15000,
                    print_progress=False,
                    strategy=SimpleGAforSyncData(
                            domain=data.domain,
                            data_size=SYNC_DATA_SIZE,
                            population_size=100,
                            elite_size=5,
                            muta_rate=1,
                            mate_rate=1,
        )
    )
    # rap = RelaxedProjection(domain=data.domain, data_size=1000, iterations=1000, learning_rate=0.05,
    #                         print_progress=False)
    delta = 1.0 / len(data) ** 2
    # Generate differentially private synthetic data with ADAPTIVE mechanism
    for eps in [0.07, 0.23, 0.52, 0.74, 1.0]:
        for seed in [0, 1, 2]:
            key = jax.random.PRNGKey(seed)
            t0 = timer()
            sync_data_2 = priv_ga.fit_dp(key, stat_module=marginal_module, epsilon=eps, delta=delta,)
            errors = jnp.abs(true_stats - stat_fn(sync_data_2.to_numpy()))
            print(f'PrivGA: eps={eps:.2f}, seed={seed}'
                  f'\t max error = {errors.max():.5f}'
                  f'\t avg error = {errors.mean():.5f}'
                  f'\t time = {timer() - t0}:.4f')
            plot_2d_data(sync_data_2.to_numpy())
        print()

