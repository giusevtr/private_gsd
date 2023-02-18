import jax.random
import matplotlib.pyplot as plt

from models import PrivGA, SimpleGAforSyncData
from stats import ChainedStatistics, Halfspace, Marginals
# from utils.utils_data import get_data
import jax.numpy as jnp
# from dp_data.data import get_data
from dp_data import load_domain_config, load_df, get_evaluate_ml
from utils import timer, Dataset, Domain

if __name__ == "__main__":
    dataset_name = 'folktables_2018_multitask_NY'
    root_path = '../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)
    df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
    df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')

    domain = Domain.fromdict(config)
    data = Dataset(df_train, domain)
    targets = ['JWMNP', 'PINCP', 'ESR', 'MIG', 'PUBCOV']

    # Get Data
    # data_name = f'folktables_2018_CA'
    # data = get_classification(DATA_SIZE=1000)

    ml_eval_fn = get_evaluate_ml(df_test, config, targets=targets, models=['LogisticRegression'])

    results = ml_eval_fn(df_train , 0)
    print(results)

    # Create statistics and evaluate
    # TODO: Choose marginals with target labels.
    key = jax.random.PRNGKey(0)
    prefix_module = Halfspace.get_kway_random_halfspaces(data.domain, k=1, rng=key, random_hs=1000)
    marginal_module2 = Marginals.get_kway_categorical(data.domain, k=2)
    stat_module = ChainedStatistics([
                                        marginal_module2,
                                         prefix_module,
                                         ])
    stat_module.fit(data)

    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module._get_workload_fn()


    features = []
    for f in data.domain.attrs:
        if f not in targets:
            features.append(f)
    train_df = data.df[features]

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

            ml_eval_fn(data.df, 0)
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

