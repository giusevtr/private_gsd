import jax.random
import jax.numpy as jnp
from models import RelaxedProjection
from stats import Marginals
from utils.utils_data import get_data
from utils.utils_data import Dataset
import time


if __name__ == "__main__":

    # Get Data
    ROUNDS = 1

    task = 'income'
    state = 'CA'
    data_name = f'folktables_2018_{task}_{state}'

    data = get_data(f'folktables_datasets/{data_name}-mixed-train',
                    domain_name=f'folktables_datasets/domain/{data_name}-num',  root_path='../../data_files')
    data, real_cols_range = data.normalize_real_values()

    # Create statistics and evaluate

    marginal_module, kway_combinations = Marginals.get_all_kway_mixed_combinations(data.domain, k_disc=0, k_real=2,
                                                                                bins=[2, 4, 8, 16, 32])
    marginal_module.fit(data)

    data_disc = data.discretize(num_bins=32)
    train_stats_module = Marginals.get_all_kway_combinations(data_disc.domain, 2)
    train_stats_module.fit(data_disc)

    rap = RelaxedProjection(domain=data_disc.domain, data_size=1000, iterations=100, learning_rate=0.05, print_progress=True)

    # Generate differentially private synthetic data with ADAPTIVE mechanism
    key = jax.random.PRNGKey(0)
    stime = time.time()
    sync_data_disc = rap.fit_dp_adaptive(key, stat_module=train_stats_module, rounds=ROUNDS,
                                 epsilon=1, delta=1e-6, tolerance=0.0, print_progress=False)
    sync_data = Dataset.to_numeric(sync_data_disc, data.domain.get_numeric_cols())

    true_stats = marginal_module.get_true_statistics()
    sync_stats = marginal_module.private_statistics_fn(sync_data)
    print(f'RAP: max error = {jnp.abs(true_stats - sync_stats).max():.5f}, '
          f'ave error = {jnp.linalg.norm(true_stats - sync_stats, ord=1) / true_stats.shape[0]:.7f}\t'
          f'time = {time.time() - stime:.5f}')

