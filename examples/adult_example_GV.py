import jax.random

from examples.run_example import generate_private_SD, run_experiment
from models_v2 import PrivGA, RelaxedProjection
from stats_v2 import Marginals, TwoWayPrefix
from utils.utils_data import get_data

import pdb

if __name__ == "__main__":

    # Get Data
    ROUNDS = 7
    data = get_data('adult', 'adult-small', root_path='../data_files/')
    # ROUNDS=15
    # data = get_data('adult', 'adult', root_path='../data_files/')

    # Create statistics and evaluate
    marginal_module = Marginals.get_all_kway_combinations(data.domain, k=3)
    # marginal_module = Marginals(data.domain, [('capital-gain', 'capital-loss'), ('sex', 'capital-loss')])
    marginal_module.fit(data)
    reg_marginal_module = Marginals.get_all_kway_combinations(data.domain, k=1)

    # Choose algorithm parameters
    priv_ga = PrivGA(
                     popsize=600,
                    top_k=20,
                    num_generations=350,
                    stop_loss_time_window=50,
                    print_progress=False,
                    start_mutations=32,
                     data_size=200,
                     regularization_statistics=reg_marginal_module
                     )
    # Generate differentially private synthetic data with ADAPTIVE mechanism
    key = jax.random.PRNGKey(0)
    sync_data_2 = priv_ga.fit_dp_adaptive(key, stat_module=marginal_module,
                     rounds=ROUNDS,
                                 epsilon=1, delta=1e-6)
    errros = marginal_module.get_sync_data_errors(sync_data_2.to_numpy())
    print(f'PrivGA: max error = {errros.max():.5f}')