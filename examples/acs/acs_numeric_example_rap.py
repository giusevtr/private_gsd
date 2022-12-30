import jax.random
from models import RelaxedProjection
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


    data_disc = data.discretize(num_bins=BINS)
    train_stats_module = Marginals.get_all_kway_combinations(data_disc.domain, 2)
    train_stats_module.fit(data_disc)

    rap = RelaxedProjection(domain=data_disc.domain, data_size=100, iterations=5000, learning_rate=0.05, print_progress=False)


    # Generate differentially private synthetic data with ADAPTIVE mechanism
    key = jax.random.PRNGKey(0)
    sync_data_2 = rap.fit_dp_adaptive(key, stat_module=train_stats_module, rounds=ROUNDS,
                                 epsilon=0.07, delta=1e-6, tolerance=0.0, print_progress=False)
    errros = marginal_module.get_sync_data_errors(sync_data_2.to_numpy())
    print(f'PrivGA: max error = {errros.max():.5f}')

