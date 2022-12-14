import jax.random

from examples.run_example import generate_private_SD, run_experiment
from models_v2 import PrivGA, RelaxedProjection
from stats_v2 import Marginals, TwoWayPrefix
from utils.utils_data import get_data

import pdb

if __name__ == "__main__":

    # Get Data
    ROUNDS=4
    data = get_data('adult', 'adult-mini', root_path='../data_files/')
    # ROUNDS=15
    # data = get_data('adult', 'adult', root_path='../data_files/')

    # Create statistics and evaluate
    marginal_module = Marginals.get_all_kway_combinations(data.domain, k=2)
    marginal_module.fit(data)

    # print(f'num queries = {marginal_module.get_num_queries()}')
    #
    # rap = RelaxedProjection(domain=data.domain,
    #                         data_size=200,
    #                         iterations=5000,
    #                         learning_rate=0.005,
    #                         print_progress=False,
    #                         )
    #
    # key = jax.random.PRNGKey(0)
    # sync_data_2 = rap.fit_dp_adaptive(key, stat_module=marginal_module,
    #                  rounds=ROUNDS, epsilon=1, delta=1e-6)
    # errros = marginal_module.get_sync_data_errors(sync_data_2.to_numpy())
    # print(f'RAP: max error = {errros.max():.5f}')


    # Choose algorithm parameters
    priv_ga = PrivGA(domain=data.domain,
                     popsize=600,
                    top_k=20,
                    num_generations=350,
                    stop_loss_time_window=50,
                    print_progress=False,
                    start_mutations=32,
                     data_size=200
                     )

    # Generate differentially private synthetic data with ONE-SHOT mechanism
    # sync_data_1 = priv_ga.fit_privately(stat_module=marginal_module,
    #                  seed=0, epsilon=1)

    # Generate differentially private synthetic data with ADAPTIVE mechanism
    key = jax.random.PRNGKey(0)
    sync_data_2 = priv_ga.fit_dp_adaptive(key, stat_module=marginal_module,
                     rounds=ROUNDS, epsilon=1, delta=1e-6)
    errros = marginal_module.get_sync_data_errors(sync_data_2.to_numpy())
    print(f'PrivGA: max error = {errros.max():.5f}')

