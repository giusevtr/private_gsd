import jax.random
import matplotlib.pyplot as plt
import pandas as pd

from models import PrivGA, SimpleGAforSyncData, RelaxedProjectionPP
from stats import ChainedStatistics, Halfspace, HalfspaceDiff, Prefix
# from utils.utils_data import get_data
from utils import timer
import jax.numpy as jnp
# from dp_data.data import get_data
from dp_data import load_domain_config, load_df, get_evaluate_ml
from utils import timer, Dataset, Domain
import seaborn as sns

def visualize(df_real, df_sync, msg=''):
    domain = Domain.fromdict(config)

    for num_col in domain.get_numeric_cols():
        real_mean = df_real[num_col].mean()
        real_std = df_real[num_col].std()
        sync_mean = df_sync[num_col].mean()
        sync_std = df_sync[num_col].std()
        print(f'{num_col:<10}. real.mean={real_mean:<5.3}, real.std={real_std:<5.3}'
              f'sync.mean={sync_mean:<5.3f}, sync.std={sync_std:<5.3f}')

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

if __name__ == "__main__":
    module = 'Prefix'
    dataset_name = 'folktables_2018_real_CA'
    root_path = '../../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)
    df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
    df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')

    halfspace_samples = 10000


    print(f'train size: {df_train.shape}')
    print(f'test size:  {df_test.shape}')
    domain = Domain.fromdict(config)
    data = Dataset(df_train, domain)

    # Create statistics and evaluate
    key = jax.random.PRNGKey(0)
    module = HalfspaceDiff.get_kway_random_halfspaces(data.domain, k=1, rng=key, random_hs=halfspace_samples)
    stat_module = ChainedStatistics([module])
    stat_module.fit(data)

    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module._get_workload_fn()

    algo = RelaxedProjectionPP(domain=data.domain, data_size=1000, iterations=1000, learning_rate=[0.01], print_progress=False)
    # Choose algorithm parameters

    delta = 1.0 / len(data) ** 2
    # Generate differentially private synthetic data with ADAPTIVE mechanism
    for eps in [1.00]:
    # for eps in [0.07, 0.23, 0.52, 0.74, 1.00]:
        # for seed in [0, 1, 2]:
        for seed in [0]:
            key = jax.random.PRNGKey(seed)
            t0 = timer()

            def debug(i, sync_data: Dataset):
                print(f'results epoch {i}:')
                results = ml_eval_fn(sync_data.df, seed)
                results = results[results['Eval Data'] == 'Test']
                print(results)
                n_sync = len(sync_data.df)
                visualize(df_real=df_train.sample(n=n_sync), df_sync=sync_data.df, msg=f'epoch={i}')

            sync_data = algo.fit_dp_adaptive(key, stat_module=stat_module, epsilon=eps, delta=delta,
                                             rounds=30,
                                             num_sample=20,
                                             debug_fn=debug
                                    )

            sync_data.df.to_csv(f'{dataset_name}_sync_{eps:.2f}_{seed}.csv', index=False)
            errors = jnp.abs(true_stats - stat_fn(sync_data.to_numpy()))
            print(f'RAP++: eps={eps:.2f}, seed={seed}'
                  f'\t max error = {errors.max():.5f}'
                  f'\t avg error = {errors.mean():.5f}'
                  f'\t time = {timer() - t0:.4f}')
            print('Final ML Results:')
            debug(-1, sync_data)

        print()

