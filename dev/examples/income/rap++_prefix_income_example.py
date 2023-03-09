import jax.random
import matplotlib.pyplot as plt
import pandas as pd
import os
from models import PrivGA, SimpleGAforSyncData, RelaxedProjectionPP
from stats import ChainedStatistics, Halfspace, HalfspaceDiff, Prefix, PrefixDiff, MarginalsDiff
# from utils.utils_data import get_data
from utils import timer
import jax.numpy as jnp
# from dp_data.data import get_data
from dp_data import load_domain_config, load_df, get_evaluate_ml
from utils import timer, Dataset, Domain , get_Xy, filter_outliers
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression



def visualize(df_real, df_sync, msg=''):
    domain = Domain.fromdict(config)

    for num_col in domain.get_numeric_cols():
        real_mean = df_real[num_col].mean()
        real_std = df_real[num_col].std()
        sync_mean = df_sync[num_col].mean()
        sync_std = df_sync[num_col].std()
        print(f'{num_col:<10}. real.mean={real_mean:<5.3}, real.std={real_std:<5.3}, '
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


def ml_eval_fn(df_test, features, target, domain):

    def fn(df_train):
        X_train, y_train, X_test, y_test = get_Xy(domain, features=features, target=target, df_train=df_train,
                                                  df_test=df_test, rescale=True)
        clf = LogisticRegression(max_iter=5000, random_state=0)
        clf.fit(X_train, y_train)
        rep = classification_report(y_test, clf.predict(X_test), output_dict=True)
        f1 = rep['macro avg']['f1-score']
        return f1

    return fn

if __name__ == "__main__":
    clipped = False
    dataset_name = 'folktables_2018_income_NY'

    root_path = '../../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)
    df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
    df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')

    domain = Domain.fromdict(config)
    data = Dataset(df_train, domain)
    data = data.drop(['OCCP', 'POBP'])
    domain = data.domain
    if clipped:
        dataset_name = dataset_name + '_clipped'
        df_train, df_test = filter_outliers(df_train, df_test, config, quantile=0.03)

    # Create statistics and evaluate
    key = jax.random.PRNGKey(0)
    # module0 = MarginalsDiff.get_all_kway_categorical_combinations(data.domain, k=2)
    # module1 = PrefixDiff.get_kway_prefixes(data.domain, k_cat=1, k_num=2, rng=key, random_prefixes=10000)
    module1 = PrefixDiff(data.domain, k_cat=1, cat_kway_combinations=[('PINCP', )], rng=key, k_prefix=2, num_random_prefixes=50000)

    stat_module = ChainedStatistics([
        # module0,
                                     module1
                                     ])
    stat_module.fit(data)

    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module.get_dataset_statistics_fn()

    algo = RelaxedProjectionPP(domain=data.domain, data_size=1000, iterations=5000, learning_rate=[0.0003], print_progress=True)

    num_sample = 10
    delta = 1.0 / len(data) ** 2
    for eps in [1.00]:
    # for eps in [0.07, 0.23, 0.52, 0.74, 1.0]:
        for seed in [0]:
        # for seed in [0, 1, 2]:
            for rounds in [50]:
                key = jax.random.PRNGKey(seed)
                t0 = timer()
                sync_dir = f'sync_data/{dataset_name}/RAP++/Prefix/{rounds}/{num_sample}/{eps:.2f}/'
                os.makedirs(sync_dir, exist_ok=True)
                sync_data = algo.fit_dp_adaptive(key, stat_module=stat_module,
                            # oneshot_stats_ids=[0],
                            # oneshot_share=0.4,
                            rounds=rounds,
                            epsilon=eps, delta=delta,
                            num_sample=num_sample,
                            )
                sync_data.df.to_csv(f'{sync_dir}/sync_data_{seed}.csv', index=False)
                errors = jnp.abs(true_stats - stat_fn(sync_data))
                print(f'RAP++({dataset_name}): eps={eps:.2f}, seed={seed}'
                      f'\t max error = {errors.max():.5f}'
                      f'\t avg error = {errors.mean():.5f}'
                      f'\t time = {timer() - t0:.4f}')

        print()

