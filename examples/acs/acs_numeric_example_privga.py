import jax.random
from models import PrivGA, SimpleGAforSyncData
from stats import Marginals
from utils.utils_data import get_data


if __name__ == "__main__":

    # Get Data
    ROUNDS = 1
    BINS=30

    task = 'coverage'
    state = 'CA'
    data_name = f'folktables_{task}_2018_{state}'
    # data = get_data(f'folktables_datasets/{data_name}-mixed',
    #                 domain_name=f'folktables_datasets/{data_name}-cat',  root_path='../../data_files/')

    # data = get_data(f'folktables_datasets/{data_name}-mixed',
    #                 domain_name=f'folktables_datasets/{data_name}-num',  root_path='../../data_files/')

    data = get_data(f'folktables_datasets/{data_name}-mixed',
                    domain_name=f'folktables_datasets/{data_name}-num',  root_path='../../data_files/')

    # Create statistics and evaluate
    marginal_module = Marginals.get_all_kway_combinations(data.domain, k=2, bins=BINS)
    marginal_module.fit(data)
    ########
    # PrivGA
    ########
    data_size = 1000
    priv_ga = PrivGA(
        num_generations=10000,
        stop_loss_time_window=50,
        print_progress=True,
        strategy=SimpleGAforSyncData(
            domain=data.domain,
            data_size=data_size,
            population_size=100,
            elite_size=10,
            muta_rate=1,
            mate_rate=10
        )
    )

    # Generate differentially private synthetic data with ADAPTIVE mechanism
    key = jax.random.PRNGKey(0)
    sync_data_2 = priv_ga.fit_dp_adaptive(key, stat_module=marginal_module, rounds=ROUNDS,
                                 epsilon=0.07, delta=1e-6, tolerance=0.0, print_progress=True)
    errros = marginal_module.get_sync_data_errors(sync_data_2.to_numpy())
    print(f'PrivGA: max error = {errros.max():.5f}')

