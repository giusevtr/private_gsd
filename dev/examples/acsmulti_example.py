import jax.random
import matplotlib.pyplot as plt

from models import PrivGA, SimpleGAforSyncData
from stats import ChainedStatistics, Halfspace, Marginals
# from utils.utils_data import get_data
import jax.numpy as jnp
# from dp_data.data import get_data
from dp_data import load_domain_config, load_df
from utils import timer, Dataset, Domain

if __name__ == "__main__":
    dataset_name = 'folktables_2018_multitask_NY'
    root_path = '../../dp-data-dev/run/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)
    df = load_df(dataset_name, root_path=root_path)

    domain = Domain.fromdict(config)
    data = Dataset(df, domain)
    # Get Data
    # data_name = f'folktables_2018_CA'
    # data = get_classification(DATA_SIZE=1000)

    # Create statistics and evaluate
    key = jax.random.PRNGKey(0)
    # prefix_module = Prefix.get_kway_prefixes(data.domain, k_cat=1, k_num=2, rng=key, random_prefixes=1000)
    prefix_module = Halfspace.get_kway_random_halfspaces(data.domain, k=1, rng=key, random_hs=1000)
    marginal_module2 = Marginals.get_kway_categorical(data.domain, k=2)
    stat_module = ChainedStatistics([
                                        marginal_module2,
                                         prefix_module,
                                         ])
    stat_module.fit(data)

    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module._get_workload_fn()

    # algo = RelaxedProjection(domain=data.domain,
    #                         data_size=1000, iterations=1000, learning_rate=0.01,)
    # Choose algorithm parameters
    SYNC_DATA_SIZE = 2000
    algo = PrivGA(num_generations=12000,
                    print_progress=False, stop_early=True,
                    strategy=SimpleGAforSyncData(domain=data.domain, elite_size=5, data_size=SYNC_DATA_SIZE))

    delta = 1.0 / len(data) ** 2
    # Generate differentially private synthetic data with ADAPTIVE mechanism
    for eps in [1.00]:
    # for eps in [0.07, 0.23, 0.52, 0.74, 1.0]:
        # for seed in [0, 1, 2]:
        for seed in [0]:
            key = jax.random.PRNGKey(seed)
            t0 = timer()
            sync_data = algo.fit_dp_adaptive(key, stat_module=stat_module, epsilon=eps, delta=delta,
                                             rounds=50,
                                             num_sample=1
                                    )
            sync_data.df.to_csv(f'{dataset_name}_sync_{eps:.2f}_{seed}.csv', index=False)
            errors = jnp.abs(true_stats - stat_fn(sync_data.to_numpy()))
            print(f'PrivGA: eps={eps:.2f}, seed={seed}'
                  f'\t max error = {errors.max():.5f}'
                  f'\t avg error = {errors.mean():.5f}'
                  f'\t time = {timer() - t0:.4f}')

            x = list(range(len(algo.fitness_record)))
            plt.plot(x, algo.fitness_record)
            plt.ylabel('Fitness')
            plt.xlabel('Iterations')
            plt.show()
            noisy_stats = stat_module.get_selected_noised_statistics()
            sel_true_stats = stat_module.get_selected_statistics_without_noise()
            gau_errors = jnp.abs(sel_true_stats - noisy_stats)
            print(  f'\t Gau max error = {gau_errors.max():.5f}')
            # plot_2d_data(sync_data.to_numpy())

        print()

