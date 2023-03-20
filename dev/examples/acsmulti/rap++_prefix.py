import jax.random
import matplotlib.pyplot as plt
import pandas as pd
import os
from models import PrivGA, SimpleGAforSyncData, RelaxedProjectionPP
from stats import ChainedStatistics, Halfspace, HalfspaceDiff, Prefix, MarginalsDiff, PrefixDiff
# from utils.utils_data import get_data
from utils import timer
import jax.numpy as jnp
# from dp_data.data import get_data
from utils import timer, Dataset, Domain , get_Xy, filter_outliers
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression


from dev.dataloading.data_functions.acs import get_acs_all

# def visualize(df_real, df_sync, msg=''):
#     domain = Domain.fromdict(config)
#
#     for num_col in domain.get_numeric_cols():
#         real_mean = df_real[num_col].mean()
#         real_std = df_real[num_col].std()
#         sync_mean = df_sync[num_col].mean()
#         sync_std = df_sync[num_col].std()
#         print(f'{num_col:<10}. real.mean={real_mean:<5.3}, real.std={real_std:<5.3}, '
#               f'sync.mean={sync_mean:<5.3f}, sync.std={sync_std:<5.3f}')
#
#         col_real = df_real[num_col].to_frame()
#         col_real['Type'] = 'real'
#         col_sync = df_sync[num_col].to_frame()
#         col_sync['Type'] = 'sync'
#
#         df = pd.concat([col_real, col_sync])
#
#         g = sns.FacetGrid(df,  hue='Type')
#         g.map(plt.hist, num_col, alpha=0.5)
#         g.fig.subplots_adjust(top=0.9)  # adjust the Figure in rp
#         g.fig.suptitle(msg)
#         g.add_legend()
#         # g.set(yscale='log')
#         plt.show()
#
#     return df_train, df_test


if __name__ == "__main__":
    dataset_name = 'folktables_2018_multitask_NY'
    data_all, data_container_fn = get_acs_all(state='NY')

    domain = data_all.domain
    cat_cols = domain.get_categorical_cols()
    num_cols = domain.get_numeric_cols()

    data_container = data_container_fn(seed=0)
    data = data_container.train

    targets = ['PINCP',  'PUBCOV', 'ESR', 'MIG', 'JWMNP']
    # Create statistics and evaluate
    key = jax.random.PRNGKey(0)
    module0 = MarginalsDiff.get_all_kway_categorical_combinations(data.domain, k=2)
    module1 = PrefixDiff(domain, k_cat=1,
                         cat_kway_combinations=[(cat, ) for cat in targets],
                         rng=key,
                         k_prefix=2, num_random_prefixes=20000)
    # module1 = HalfspaceDiff(domain=data.domain, k_cat=1,
    #                         cat_kway_combinations=[('PINCP',),  ('PUBCOV', ), ('ESR', )], rng=key,
    #                         num_random_halfspaces=200000)
    stat_module = ChainedStatistics([module0,
                                     module1
                                     ])
    stat_module.fit(data)

    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module.get_dataset_statistics_fn()

    algo = RelaxedProjectionPP(domain=data.domain, data_size=1000,
                               iterations=1000, learning_rate=[0.01], print_progress=False)

    num_sample = 10
    delta = 1.0 / len(data) ** 2
    for seed in [0]:
        # for seed in [0, 1, 2]:
        for eps in [1.00]:
    # for eps in [0.07, 0.23, 0.52, 0.74, 1.00, 10.0]:
            for rounds in [50]:
                num_adaptive_queries = rounds * num_sample
                oneshot_share = module0.get_num_workloads() / (module0.get_num_workloads() + num_adaptive_queries)
                print(f'oneshot_share={oneshot_share:.4f}')

                key = jax.random.PRNGKey(seed)
                t0 = timer()
                sync_dir = f'sync_data/{dataset_name}/RAP++/Prefix/{rounds}/{num_sample}/{eps:.2f}/'
                os.makedirs(sync_dir, exist_ok=True)
                sync_data = algo.fit_dp_hybrid(key, stat_module=stat_module,
                            oneshot_stats_ids=[0],
                            oneshot_share=oneshot_share,
                            rounds=rounds,
                            epsilon=eps, delta=delta,
                            num_sample=num_sample,
                            )
                sync_data.df.to_csv(f'{sync_dir}/sync_data_{seed}.csv', index=False)
                errors = jnp.abs(true_stats - stat_fn(sync_data))
                print(f'RAP++: eps={eps:.2f}, seed={seed}'
                      f'\t max error = {errors.max():.5f}'
                      f'\t avg error = {errors.mean():.5f}'
                      f'\t time = {timer() - t0:.4f}')

        print()

