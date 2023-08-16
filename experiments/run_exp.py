from stats.get_marginals_fn_v2 import get_marginal, Query
import pandas as pd
import os
from utils import Dataset, Domain, timer
import jax.random
from models import GSD
import jax.numpy as jnp
# from stats.get_marginals_fn import get_marginal_query
from experiments.utils_for_experiment import read_original_data, read_tabddpm_data
from utils import MLEncoder
from dp_data import cleanup, DataPreprocessor, get_config_from_json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier
score_fn = make_scorer(f1_score, average='macro')
get_lg_function = lambda rs: LogisticRegression(random_state=rs)
get_ml_function = lambda rs: XGBClassifier(random_state=rs)

def get_ml_score_fn(data_name):
    train_df, test_df, all_df, cat_cols, ord_cols, real_cols = read_original_data(data_name, root_dir='data2/data')
    encoder = MLEncoder(cat_features=cat_cols, num_features=ord_cols + real_cols,
                        target='Label', rescale=False)
    print(f'{data_name}')
    print(f'Categorical columns =',cat_cols)
    print(f'Ordinal columns =', ord_cols)
    print(f'Real-valued columns =', real_cols)
    encoder.fit(all_df)
    X_test_oh, y_test = encoder.encode(test_df)

    def test_score_fn(sync_df, protected_feature=None):
        X_sync_oh, y_sync = encoder.encode(sync_df)
        model_sync = get_ml_function(0)
        model_sync.fit(X_sync_oh, y_sync)
        sync_train = score_fn(model_sync, X_sync_oh, y_sync)
        sync_test = score_fn(model_sync, X_test_oh, y_test)
        if protected_feature is not None:
            subgroup = subgroup_evaluation(model_sync, encoder, test_df, protected_feature)
            return sync_train, sync_test, subgroup
        return sync_train, sync_test

    original_train, original_test = test_score_fn(train_df)
    print(f'\tOriginal:\t{original_train:.5f}\t{original_test:.5f}')

    return test_score_fn


def subgroup_evaluation(model, encoder, test_df, protected_feat):
    groups = test_df[protected_feat].unique()
    group_val = dict([(g, []) for g in groups])
    res = []
    for g in groups:
        group_df = test_df[test_df[protected_feat] == g]
        X_sub_oh, y_sub = encoder.encode(group_df)
        subgroup_test = score_fn(model, X_sub_oh, y_sub)
        group_val[g].append(subgroup_test)
        print(f'\t\tgroup={g:<20}\t\t{subgroup_test:.5f}')

        res.append((g, subgroup_test))
    res_df = pd.DataFrame(res, columns=['G', 'Subgroup Test F1'])
    return res_df


def ml_evaluation(data_name, data_size_str='N',
                  stat_name='k',
                min_density=0.99,
                max_marginal_size=1000,
                   protected_feature=None, seeds=(0,)):

    orig_train_df, ml_score_fn = get_ml_score_fn(data_name)

    orig_train, orig_test = ml_score_fn(orig_train_df, protected_feature=protected_feature)

    for seed in list(seeds):
        sync_dir = f'sync_data/{data_name}/{data_size_str}/{stat_name}/{max_marginal_size}/{min_density:.3f}/inf/{seed}'
        print(f'reading {sync_dir}')
        sync_df = pd.read_csv(f'{sync_dir}/sync_data.csv').dropna()
        sync_train, sync_test = ml_score_fn(sync_df, protected_feature=protected_feature)
        print(f'\tSync:\t{sync_train:.5f}\t{sync_test:.5f}')



def run_experiment(data_name, data_size_str='N',
                   protected_feature=None, seeds=(0,), verbose=False):

    print('ML Score:')
    ml_score_fn = get_ml_score_fn(data_name)

    train_df, test_df, all_df, cat_cols, ord_cols, real_cols = read_original_data(data_name, root_dir='data2/data')
    print(f'Categorical columns =',cat_cols)
    print(f'Ordinal columns =', ord_cols)
    print(f'Real-valued columns =', real_cols)


    config = get_config_from_json({'categorical': cat_cols + ['Label'], 'ordinal': ord_cols, 'numerical': real_cols})
    preprocessor = DataPreprocessor(config=config)
    preprocessor.fit(all_df)
    pre_train_df = preprocessor.transform(train_df)

    N = len(train_df)
    N_prime = N if data_size_str == 'N' else int(data_size_str)

    # Create dataset and k-marginal queries
    config_dict = preprocessor.get_domain()
    domain = Domain(config_dict)
    data = Dataset(pre_train_df, domain)

    algo = GSD(num_generations=200000,
                   print_progress=verbose,
                   stop_early=True,
                   domain=data.domain,
                   population_size_muta=20,
                    population_size_cross=20,
                   data_size=N_prime,
                   stop_early_gen=N_prime,
                   )
        # delta = 1.0 / len(data) ** 2
    for seed in list(seeds):
        sync_dir = f'sync_data/{data_name}/{data_size_str}/inf/{seed}'
        os.makedirs(sync_dir, exist_ok=True)

        key = jax.random.PRNGKey(seed)

        t0 = timer()
        include_features = ['Label']
        if protected_feature is not None and protected_feature in domain.attrs:
            include_features.append(protected_feature)

        true_stats, stat_fn = get_marginal(data, maximum_size=500000, conditional_col=include_features, verbose=verbose)
        # true_stats, stat_fn, total_error_fn = get_marginal_query(seed, data, domain, k=k,
        #                                                          min_bin_density=0.005,
        #                                                          minimum_density=min_density,
        #                                                          max_marginal_size=max_marginal_size,
        #                                                          min_marginal_size=1000,
        #                                                          include_features=include_features, verbose=verbose)
        # marginal_query = Query(data, maximum_size=200000, conditional_col=include_features, verbose=verbose)
        # true_stats = marginal_query.get_true_stats()
        ml_progress = []
        def debug_fn(epoch, fitness, sync_data):
            # Track ML error
            fitness = float(fitness)
            sync_data_df = sync_data.df.copy()
            sync_data_df_post: pd.DataFrame
            sync_data_df_post = preprocessor.inverse_transform(sync_data_df)
            sync_data_df_post_upsampled = sync_data_df_post.sample(n=N, replace=True)

            sync_train, sync_test = ml_score_fn(sync_data_df_post_upsampled)
            print(f'\tgeneration={epoch}: Fit={fitness:.6f} Sync:\t{sync_train:.5f}\t{sync_test:.5f}')
            ml_progress.append((epoch, fitness, sync_train, sync_test))
            progress_df = pd.DataFrame(ml_progress, columns=['G', 'Fitness', 'Train', 'Test'])
            progress_df.to_csv(f'{sync_dir}/ml_progress.csv')


        sync_data = algo.fit_help(key, true_stats, stat_fn, debug_fn=debug_fn)
        # sync_data = algo.fit_ada_non_priv(key, true_stats, marginal_query, rounds=10, samples_per_round=2000, debug_fn=debug_fn)
        sync_data_df = sync_data.df.copy()
        sync_data_df_post = preprocessor.inverse_transform(sync_data_df)


        print(f'Saving {sync_dir}/sync_data.csv')
        sync_data_df_post.to_csv(f'{sync_dir}/sync_data.csv', index=False)
        # errors = jnp.abs(true_stats - marginal_query.get_all_stats(sync_data))
        errors = jnp.abs(true_stats - stat_fn(sync_data.to_numpy()))

        elapsed_time = int(timer() - t0)
        f = open(f"{sync_dir}/time.txt", "w")
        f.write(f'{elapsed_time}')
        f.close()

        print(f'Input data {data_name}, seed={seed}')
        print(f'GSD(oneshot):'
              f'\t max error = {errors.max():.5f}'
              f'\t avg error = {errors.mean():.6f}')

        # total_errors_df = total_error_fn(data, sync_data)
        # print('\tTotal max error = ', total_errors_df['Max'].max())
        # print('\tTotal avg error = ', total_errors_df['Average'].mean())
        # total_errors_df.to_csv(f'{sync_dir}/errors.csv')

        sync_train, sync_test = ml_score_fn(sync_data_df_post)
        print(f'\tSync:\t{sync_train:.5f}\t{sync_test:.5f}')
