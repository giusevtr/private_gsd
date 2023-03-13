import jax.random
import matplotlib.pyplot as plt

from models import PrivGA, SimpleGAforSyncData, RelaxedProjectionPP
from stats import ChainedStatistics, Prefix
# from utils.utils_data import get_data
from utils import timer
import jax.numpy as jnp
# from dp_data.data import get_data
from dp_data import load_domain_config, load_df
from utils import timer, Dataset, Domain
import pandas as pd
import seaborn as sns
import numpy as np

import os


def visualize(df_real, df_sync, msg=''):
    domain = Domain.fromdict(config)

    for num_col in domain.get_numeric_cols():
        col_real = df_real[num_col].to_frame()
        col_real['Type'] = 'real'
        col_sync = df_sync[num_col].to_frame()
        col_sync['Type'] = 'sync'

        df = pd.concat([col_real, col_sync])

        g = sns.FacetGrid(df,  hue='Type')
        g.map(plt.hist, num_col, alpha=0.5)
        g.fig.subplots_adjust(top=0.9)  # adjust the Figure in rp
        g.fig.suptitle(msg)
        g.add_legend()
        # g.set(yscale='log')
        plt.show()

    return df_train, df_test

def rescale(df, num_cols):

    for c in num_cols:
        mean = df[c].mean()
        std = df[c].std()

        temp = (df[c] - mean) / (4 * std) + 0.5
        temp_np = np.clip(temp.values, 0, 1)
        df[c] = temp_np

        df[c].hist()
        plt.title(f'Rescale {c}')
        plt.show()
        print(f'Done with {c}')

    return df

if __name__ == "__main__":
    dataset_name = 'folktables_2018_real_CA'
    root_path = '../../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)
    df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
    df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')

    print(f'train size: {df_train.shape}')
    print(f'test size:  {df_test.shape}')
    domain = Domain.fromdict(config)
    # df_train = df_train.sample(n=50000)
    halfspace_samples = 50000

    df_train = rescale(df_train, domain.get_numeric_cols())
    df_test = rescale(df_test, domain.get_numeric_cols())


    data = Dataset(df_train, domain)

    # Create statistics and evaluate
    key = jax.random.PRNGKey(0)
    module = Prefix(domain=data.domain, k_cat=1, cat_kway_combinations=[('PINCP',),  ('PUBCOV', )], rng=key, num_random_prefixes=50000)
    stat_module = ChainedStatistics([module])
    stat_module.fit(data)

    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module._get_workload_fn()

    # algo = RelaxedProjectionPP(domain=data.domain, data_size=1000, iterations=1000, learning_rate=[0.010], print_progress=False)
    # Choose algorithm parameters
    algo = PrivGA(num_generations=40000, print_progress=False, stop_early=True, strategy=SimpleGAforSyncData(domain=data.domain, elite_size=5, data_size=2000))

    rounds = 50
    samples = 10

    delta = 1.0 / len(data) ** 2
    # Generate differentially private synthetic data with ADAPTIVE mechanism
    for eps in [1.00]:
        sync_dir = f'sync_data/{dataset_name}/GSD/Prefix/oneshot/{eps:.2f}/'

        os.makedirs(sync_dir, exist_ok=True)

        # for eps in [0.07, 0.23, 0.52, 0.74, 1.0]:
        # for seed in [0, 1, 2]:
        for seed in [0]:
            key = jax.random.PRNGKey(seed)
            t0 = timer()

            sync_data = algo.fit_dp_adaptive(key, stat_module=stat_module, epsilon=eps, delta=delta,
                                             rounds=50,
                                             num_sample=10,
                                    )
            sync_data.df.to_csv(f'{dataset_name}_sync_{eps:.2f}_{seed}.csv', index=False)
            errors = jnp.abs(true_stats - stat_fn(sync_data.to_numpy()))
            print(f'PrivGA: eps={eps:.2f}, seed={seed}\t max error = {errors.max():.5f}\t avg error = {errors.mean():.5f}'
                  f'\t time = {timer() - t0:.4f}')


        print()
