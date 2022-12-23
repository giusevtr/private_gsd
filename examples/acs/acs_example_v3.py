import jax.random
from models_v3 import PrivGA
from stats_v3 import Marginals
from utils.utils_data import get_data


if __name__ == "__main__":

    # Get Data
    ROUNDS = 25

    task = 'coverage'
    state = 'CA'
    data_name = f'folktables_{task}_2018_{state}'
    data = get_data(f'folktables_datasets/{data_name}-mixed',
                    domain_name=f'folktables_datasets/{data_name}-mixed',  root_path='../../data_files/')

    # Create statistics and evaluate
    marginal_module = Marginals.get_all_kway_combinations(data.domain, k=3, bins=10)
    marginal_module.fit(data)

    # Choose algorithm parameters
    priv_ga = PrivGA(
        domain=data.domain,
                     popsize=1000,
                    top_k=2,
                    num_generations=1000,
                    stop_loss_time_window=50,
                    print_progress=False,
                    start_mutations=64,
                     data_size=200,
                     )
    # Generate differentially private synthetic data with ADAPTIVE mechanism
    key = jax.random.PRNGKey(0)
    sync_data_2 = priv_ga.fit_dp_adaptive(key, stat_module=marginal_module, rounds=ROUNDS,
                                 epsilon=0.07, delta=1e-6, tolerance=0.01, print_progress=True)
    errros = marginal_module.get_sync_data_errors(sync_data_2.to_numpy())
    print(f'PrivGA: max error = {errros.max():.5f}')

