import pandas as pd

from stats import Marginals
from utils.utils_data import get_data
import jax.numpy as jnp

from utils import Dataset


def get_subgroup(data_arg: Dataset, stat_fn,  att: str, value: int):
    df_data = data_arg.df
    df_given = df_data[(df_data[att] == value)]

    subgroup_data = Dataset(df_given, domain=data_arg.domain)
    subgroup_stats = stat_fn(subgroup_data.to_numpy())
    return subgroup_stats


if __name__ == "__main__":

    # stat_name = '2way_only'
    stat_name = '1_and_2way'
    # tasks = ['mobility', 'coverage', 'income', 'employment', 'travel']
    # Get Data
    # data_name = f'folktables_2018_CA'
    data_name = 'folktables_2018_mobility_CA'
    data = get_data(f'{data_name}-mixed-train', domain_name=f'domain/{data_name}-cat', root_path='../../../data_files/folktables_datasets')
    # data = get_classification(DATA_SIZE=1000)

    SYNC_DATA_SIZE = 2000
    # Create statistics and evaluate
    k = 2
    marginal_module, _ = Marginals.get_all_kway_combinations(data.domain, k=k, bins=[2, 4, 8, 16, 32])
    marginal_module.fit(data)

    true_stats = marginal_module.get_all_true_statistics()
    stat_fn = marginal_module._get_workload_fn()

    statistics_for_whites = get_subgroup(data, stat_fn, 'RAC1P', 0)
    statistics_for_blacks = get_subgroup(data, stat_fn, 'RAC1P', 1)

    # rap = RelaxedProjection(domain=data.domain, data_size=1000, iterations=1000, learning_rate=0.05,
    #                         print_progress=False)
    delta = 1.0 / len(data) ** 2
    # Generate differentially private synthetic data with ADAPTIVE mechanism
    Res = []
    for eps in [0.07, 0.23, 0.52, 0.74, 1.0]:
        for seed in [0, 1, 2]:

            path_dir = f'{stat_name}/{data_name}/{eps}/'
            sync_path = f'{path_dir}/sync_data_{seed}.csv'
            df = pd.read_csv(sync_path, index_col=None)

            sync_data = Dataset(df, data.domain)

            sync_statistics_for_whites = get_subgroup(sync_data, stat_fn, 'RAC1P', 0)
            sync_statistics_for_blacks = get_subgroup(sync_data, stat_fn, 'RAC1P', 1)
            errors_given_white = jnp.abs(statistics_for_whites - sync_statistics_for_whites)
            errors_given_black = jnp.abs(statistics_for_blacks - sync_statistics_for_blacks)

            errors = jnp.abs(true_stats - stat_fn(sync_data.to_numpy()))
            print(f'PrivGA: eps={eps:.2f}, seed={seed}'
                  f'\t max error = {errors.max():.5f}'
                  f'\t avg error = {errors.mean():.5f}'
                  f'\t max error|white = {errors_given_white.max():.5f}'
                  f'\t avg error|white = {errors_given_white.mean():.5f}'
                  f'\t max error|black = {errors_given_black.max():.5f}'
                  f'\t avg error|black = {errors_given_black.mean():.5f}'
                  )
            Res.append(['PrivGA', eps, seed, k, 'all', float(errors.max()), float(errors.mean())])
            Res.append(['PrivGA', eps, seed, k, 'white', float(errors_given_white.max()), float(errors_given_white.mean())])
            Res.append(['PrivGA', eps, seed, k, 'black', float(errors_given_black.max()), float(errors_given_black.mean())])

        print()
    cols = ['generator', 'epsilon', 'seed', 'k', 'subgroup', 'error_max', 'error_mean']
    res_df = pd.DataFrame(Res, columns=cols)
    res_df.to_csv(f'privga_{stat_name}_results.csv', index=False)