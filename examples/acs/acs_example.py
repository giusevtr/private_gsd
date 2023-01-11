import jax.random
from models import PrivGA, SimpleGAforSyncData
from stats import Marginals
from utils.utils_data import get_data


if __name__ == "__main__":

    # Get Data
    ROUNDS = 50

    task = 'coverage'
    state = 'CA'
    data_name = f'folktables_2018_{task}_{state}'
    data = get_data(f'folktables_datasets/{data_name}-mixed-train',
                    domain_name=f'folktables_datasets/domain/{data_name}-cat',  root_path='../../data_files/')
    SYNC_DATA_SIZE = 2000
    # Create statistics and evaluate
    marginal_module = Marginals.get_all_kway_combinations(data.domain, k=3, bins=30)
    marginal_module.fit(data)

    strategy = SimpleGAforSyncData(
            domain=data.domain,
            data_size=SYNC_DATA_SIZE,
            population_size=1000,
            elite_size=2,
            muta_rate=1,
            mate_rate=200,
        )
    # Choose algorithm parameters
    priv_ga = PrivGA(
                    num_generations=10000,
                    stop_loss_time_window=100,
                    print_progress=True,
                    strategy=strategy
    )
    # rap = RelaxedProjection(domain=data.domain, data_size=1000, iterations=1000, learning_rate=0.05,
    #                         print_progress=False)
    delta = 1.0 / len(data) ** 2
    # Generate differentially private synthetic data with ADAPTIVE mechanism
    key = jax.random.PRNGKey(0)
    sync_data_2 = priv_ga.fit_dp_adaptive(key, stat_module=marginal_module, rounds=ROUNDS,
                                 epsilon=0.07, delta=delta, tolerance=0.00, print_progress=True,start_X=True)
    errros = marginal_module.get_sync_data_errors(sync_data_2.to_numpy())
    print(f'PrivGA: max error = {errros.max():.5f}')

