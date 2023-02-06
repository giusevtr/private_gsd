import jax.random
import jax.numpy as jnp
from models import PrivGA, SimpleGAforSyncData
from stats import Marginals
from utils.utils_data import get_data
from utils.utils_data import Dataset
import time


if __name__ == "__main__":

    # Get Data
    ROUNDS = 100
    # adaptive_rounds = (3, 10, 100)

    task = 'mobility'
    state = 'CA'
    data_name = f'folktables_2018_{task}_{state}'
    data = get_data(f'folktables_datasets/{data_name}-mixed-train',
                    domain_name=f'folktables_datasets/domain/{data_name}-cat',  root_path='../../data_files/')

    # cat_cols = ['COW', 'RELP',  'RAC1P', 'SCHL']
    # data = data.project(cat_cols)
    # Create statistics and evaluate
    marginal_module = Marginals.get_all_kway_combinations(data.domain, k=3)[0]
    marginal_module.fit(data)

    # print(f'Workloads = {len(marginal_module.true_stats)}')

    ##########
    # PrivGA #
    ##########
    data_size = 2000
    priv_ga = PrivGA(
        num_generations=500000,
        print_progress=True,
        strategy=SimpleGAforSyncData(
            domain=data.domain,
            data_size=data_size,
            population_size=2000,
            elite_size=2,
            muta_rate=1,
            mate_rate=1
        )
    )

    # Generate differentially private synthetic data with ADAPTIVE mechanism
    key = jax.random.PRNGKey(2)
    stime = time.time()

    sync_data = priv_ga.fit_dp_adaptive(key, stat_module=marginal_module, rounds=ROUNDS, start_sync=True,
                                        epsilon=1.00, delta=1e-6, tolerance=0.0, print_progress=True)

    true_stats = marginal_module.get_true_stats()
    sync_stats = marginal_module.get_stats_jit(sync_data)
    print(f'PrivGA(fast): max error = {jnp.abs(true_stats - sync_stats).max():.5f}, '
          f'ave error = {jnp.linalg.norm(true_stats - sync_stats, ord=1) / true_stats.shape[0]:.7f}\t'
          f'time = {time.time() - stime:.5f}')

