import jax.random
from models import PrivGA, SimpleGAforSyncData
from stats import Marginals, ChainedStatistics, Halfspace, Prefix
from utils import timer
import jax.numpy as jnp
# from dp_data.data_preprocessor import ge
from dp_data import get_dataset, get_data

if __name__ == "__main__":
    data_name = 'folktables_2018_employment_CA'
    ROOT_PATH = '../dp-data/datasets/preprocessed/folktables/1-Year'
    IDXS_PATH = 'seed0/train'
    data = get_data(data_name,
                    # root_path=ROOT_PATH,
                    idxs_path=IDXS_PATH)


    # Get Data
    # data = get_data(f'{data_name}-mixed-train', domain_name=f'domain/{data_name}-mixed',  root_path='../data_files/folktables_datasets')
    print(f'Dataset: {data_name}')

    # Create statistics and evaluate
    marginal_module1 = Marginals.get_all_kway_combinations(data.domain, k=1, bins=[2, 4, 8, 16, 32])
    marginal_module2 = Marginals.get_all_kway_combinations(data.domain, k=2, bins=[2, 4, 8, 16, 32])
    marginal_module = ChainedStatistics([marginal_module1, marginal_module2])
    marginal_module.fit(data)

    true_stats = marginal_module.get_all_true_statistics()
    stat_fn = marginal_module._get_workload_fn()

    # Choose algorithm parameters
    algo = PrivGA(num_generations=20000, print_progress=True,
                    strategy=SimpleGAforSyncData(domain=data.domain, data_size=2000))

    delta = 1.0 / len(data) ** 2
    # Generate differentially private synthetic data with ADAPTIVE mechanism
    for eps in [1.0]:
    # for eps in [0.07, 0.23, 0.52, 0.74, 1.0]:
        # for seed in [0, 1, 2]:
        for seed in [0]:
            key = jax.random.PRNGKey(seed)
            t0 = timer()
            sync_data = algo.fit_dp(key, stat_module=marginal_module, epsilon=eps, delta=delta,)
            sync_data.df.to_csv(f'{data_name}_sync_{eps:.2f}_{seed}.csv', index=False)
            errors = jnp.abs(true_stats - stat_fn(sync_data.to_numpy()))
            print(f'PrivGA: eps={eps:.2f}, seed={seed}'
                  f'\t max error = {errors.max():.5f}'
                  f'\t avg error = {errors.mean():.5f}'
                  f'\t time = {timer() - t0:.4f}')

            noisy_stats = marginal_module.get_selected_noised_statistics()
            gau_errors = jnp.abs(true_stats - noisy_stats)
            print(  f'\t Gau max error = {gau_errors.max():.5f}')
            # plot_2d_data(sync_data.to_numpy())

        print()

