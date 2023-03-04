import jax.random
import matplotlib.pyplot as plt

from models import GeneticSD, GeneticStrategy
from stats import ChainedStatistics, Halfspace, Prefix, Marginals
# from utils.utils_data import get_data
from utils import timer
import jax.numpy as jnp
# from dp_data.data import get_data
from dp_data import load_domain_config, load_df, get_evaluate_ml
from utils import timer, Dataset, Domain

def filter_outliers(df_train, df_test):
    domain = Domain.fromdict(config)

    df = df_train.append(df_test)

    for num_col in domain.get_numeric_cols():
        q_lo = df[num_col].quantile(0.01)
        q_hi = df[num_col].quantile(0.96)
        size1 = len(df_train)
        df_train_filtered = df_train[(df_train[num_col] <= q_hi) & (df_train[num_col] >= q_lo)]
        df_test_filtered = df_test[(df_test[num_col] <= q_hi) & (df_test[num_col] >= q_lo)]
        size2 = len(df_train_filtered)
        print(f'Numeric column={num_col}. Removing {size1 - size2} rows.')
        # df_filtered = df[(df[num_col] <= q_hi)]
        df_train = df_train_filtered
        df_test = df_test_filtered

    for num_col in domain.get_numeric_cols():
        maxv = df_train[num_col].max()
        minv = df_train[num_col].min()
        meanv = df_train[num_col].mean()
        print(f'Col={num_col:<10}: mean={meanv:<5.3f}, min={minv:<5.3f},  max={maxv:<5.3f},')
        df_train[num_col].hist()
        plt.yscale('log')
        plt.show()

    return df_train, df_test

if __name__ == "__main__":
    dataset_name = 'folktables_2018_real_NY'
    root_path = '../../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)
    df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
    df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')

    # df_train, df_test = filter_outliers(df_train, df_test)
    print(f'train size: {df_train.shape}')
    print(f'test size:  {df_test.shape}')
    domain = Domain.fromdict(config)
    data = Dataset(df_train, domain)

    targets = ['PINCP',  'PUBCOV', 'ESR']
    #############
    ## ML Function
    ml_eval_fn = get_evaluate_ml(df_test, config, targets=targets, models=['LogisticRegression'])

    # Create statistics and evaluate
    key = jax.random.PRNGKey(0)
    # prefix_module = Prefix.get_kway_prefixes(data.domain, k_cat=1, k_num=2, rng=key, random_prefixes=1000)
    # module = Prefix.get_kway_prefixes(data.domain, k_cat=1, k_num=2, rng=key, random_prefixes=100000)
    # module = Halfspace.get_kway_random_halfspaces(data.domain, k=1, rng=key, random_hs=20000)
    module = Marginals.get_all_kway_mixed_combinations_v1(data.domain, k=2, bins=[2, 4, 8, 16, 32])
    # marginal_module2 = Marginals.get_all_kway_combinations(data.domain, k=2, bins=[2, 4, 8, 16, 32])
    stat_module = ChainedStatistics([module])
    stat_module.fit(data)

    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module._get_workload_fn()

    # Choose algorithm parameters
    SYNC_DATA_SIZE = 5000
    algo = GeneticSD(num_generations=30000,
                     print_progress=True, stop_early=True,
                     strategy=GeneticStrategy(domain=data.domain, elite_size=5, data_size=SYNC_DATA_SIZE))

    delta = 1.0 / len(data) ** 2
    # Generate differentially private synthetic data with ADAPTIVE mechanism
    for eps in [1.00]:
    # for eps in [0.07, 0.23, 0.52, 0.74, 1.0]:
        # for seed in [0, 1, 2]:
        for seed in [0]:
            key = jax.random.PRNGKey(seed)
            t0 = timer()

            sync_data = algo.fit_dp(key, stat_module=stat_module, epsilon=eps, delta=delta)
            results = ml_eval_fn(sync_data.df, seed)
            results = results[results['Eval Data'] == 'Test']
            print(results)


        print()

