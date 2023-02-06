import os

import jax.random
import matplotlib.pyplot as plt
import pandas as pd

from models import PrivGA, SimpleGAforSyncData
from stats import Marginals
from utils.utils_data import get_data
from utils import timer
from toy_datasets.classification import get_classification
import jax.numpy as jnp
from utils.plot_low_dim_data import plot_2d_data
from utils.cdp2adp import cdp_rho

from utils import Dataset


def get_subgroup(data_arg: Dataset, stat_fn,  att: str, value: int):
    df_data = data_arg.df
    df_given = df_data[(df_data[att] == value)]

    subgroup_data = Dataset(df_given, domain=data_arg.domain)
    subgroup_stats = stat_fn(subgroup_data.to_numpy())
    subgroup_size = len(subgroup_data.df)
    return subgroup_stats, subgroup_size

import numpy as np
def gaussian_conditional_query(data: Dataset):

    N = len(data.df)
    delta = 1 / len(data.df) ** 2
    rho = cdp_rho(0.1, delta)
    sensitivity = 1 / N
    sigma_gaussian = float(np.sqrt(sensitivity ** 2 / (2 * rho)))

    groups = data.df['RAC1P'].value_counts(normalize=True)

    group_ids = groups.index
    group_counts = groups.values

    for g_id, g_frac in zip(group_ids, group_counts):
        print(g_id)
        df_subgroup = data.df[data.df['RAC1P'] == g_id]

        gau_noise = np.random.normal(0, sigma_gaussian)

        g_frac_noised = g_frac + gau_noise
        print(f'subgroup size = {len(df_subgroup)}')
        print(f'Q(race={g_id}) = {g_frac:.5f}|\tP(race={g_id}) = {g_frac_noised:.5f}')
        for c in data.domain.attrs:
            # if c != 'RAC1P': continue
            if c != 'ESP': continue
            print(f'\tFeature ={c}')
            # cols.append((c, 'RAC1P'))
            counts = df_subgroup[c].value_counts(normalize=True)
            categories = counts.index
            categories_counts = counts.values

            for cat, cat_frac in zip(categories, categories_counts):
                df = data.df
                temp_df = df_subgroup.loc[df[c] == cat]
                and_count = len(temp_df)
                and_frac = and_count / N
                gau_noise = np.random.normal(0, sigma_gaussian)
                and_frac_noised = and_frac + gau_noise

                cond_answer = and_frac / g_frac
                cond_answer_noised = and_frac_noised / g_frac_noised
                print(
                    f'\t\tQ({c}={cat} and race={g_id})={and_frac:.4f}, Q({c}={cat}| race={g_id}) = {cond_answer:.7f}|\t\t'
                    f'\t\tP({c}={cat} and race={g_id})={and_frac_noised:.4f}, P({c}={cat}| race={g_id}) = {and_frac_noised / g_frac_noised:.7f}\t\t'
                    f'\t\tError={np.abs(cond_answer_noised - cond_answer):.7f}\t\t'
                )
            print()

    print()

def syncdata_conditional_query(data: Dataset, sync_data):

    N = len(data.df)
    N_sync = len(sync_data.df)

    true_groups = data.df['RAC1P'].value_counts(normalize=True)
    sync_groups = sync_data.df['RAC1P'].value_counts(normalize=True)

    group_ids = true_groups.index
    for g_id in group_ids:

        g_frac_true = true_groups[g_id]
        g_frac_sync = sync_groups[g_id]

        print(g_id)
        true_df_subgroup = data.df[data.df['RAC1P'] == g_id]
        sync_df_subgroup = sync_data.df[sync_data.df['RAC1P'] == g_id]

        # gau_noise = np.random.normal(0, sigma_gaussian)
        # g_frac_noised = g_frac + gau_noise


        for c in data.domain.attrs:
            if c == 'RAC1P': continue
            # if c != 'ESP': continue
            # print(f'\tFeature ={c}')
            # cols.append((c, 'RAC1P'))
            # true_counts = true_df_subgroup[c].value_counts(normalize=True)
            # sync_counts = sync_df_subgroup[c].value_counts(normalize=True)
            # categories = true_counts.index
            # categories_counts = counts.values

            categories = data.df[c].unique()
            for cat in categories:
                true_and_frac = len(true_df_subgroup.loc[true_df_subgroup[c] == cat]) / N
                sync_and_frac = len(sync_df_subgroup.loc[sync_df_subgroup[c] == cat]) / N_sync


                cond_answer = true_and_frac / g_frac_true
                cond_answer_noised = sync_and_frac / g_frac_sync
                error = np.abs(cond_answer_noised - cond_answer)
                if error>0.03:

                    print(f'subgroup size = {len(true_df_subgroup)}')
                    print(f'Q(race={g_id}) = {g_frac_true:.5f}|\tP(race={g_id}) = {g_frac_sync:.5f}')
                    print(
                        f'\t\tQ({c:>5}={cat:<3} and race={g_id})={true_and_frac:.4f}, Q({c:>5}={cat:<3}| race={g_id}) = {cond_answer:.7f}|\t\t'
                        f'\t\tP({c:>5}={cat:<3} and race={g_id})={sync_and_frac:.4f}, P({c:>5}={cat:<3}| race={g_id}) = {cond_answer_noised:.7f}\t\t'
                        f'\t\tError={error:.7f}\t\t'
                    )
                    print()
            print()

    print()
if __name__ == "__main__":

    tasks = ['mobility', 'coverage', 'income', 'employment', 'travel']
    # Get Data
    # data_name = f'folktables_2018_CA'
    task = 'coverage'
    data_name = f'folktables_2018_{task}_CA'
    data = get_data(f'{data_name}-mixed-train', domain_name=f'domain/{data_name}-cat',  root_path='../../data_files/folktables_datasets')

    # gaussian_conditional_query(data)

    sync_path = f'sync_folktables_2018_coverage_CA.csv'

    # df = pd.read_csv(sync_path, index_col=None)
    # sync_data = Dataset(df, data.domain)
    # syncdata_conditional_query(data, sync_data)



    marginal_module, _ = Marginals.get_all_kway_combinations(data.domain, k=2, bins=[2, 4, 8, 16, 32])
    marginal_module.fit(data)
    true_stats = marginal_module.get_all_true_statistics()
    stat_fn = marginal_module._get_workload_fn()
    # statistics_for_majority = get_subgroup(data, stat_fn, 'RAC1P', 0)

    g_size = []
    g_error = []
    for subgroup_id in [0, 1, 2, 3, 4, 5, 6, 7]:
        SYNC_DATA_SIZE = 2000
        k = 1

        true_subgroup_statistics, subgroup_size = get_subgroup(data, stat_fn, 'RAC1P', subgroup_id)
        df = pd.read_csv(sync_path, index_col=None)
        sync_data = Dataset(df, data.domain)
        sync_subgroup_statistics, _ = get_subgroup(sync_data, stat_fn, 'RAC1P', subgroup_id)
        subgroup_errors = jnp.abs(true_subgroup_statistics - sync_subgroup_statistics)

        errors = jnp.abs(true_stats - stat_fn(sync_data.to_numpy()))
        print(f'PrivGA: '
              f'\t max error = {errors.max():.5f}'
              f'\t avg error = {errors.mean():.5f}'
              f'\t subgroup({subgroup_id}) with size {subgroup_size:<5}; '
              f'\tmax error = {subgroup_errors.max():.5f}'
              f'\tavg error = {subgroup_errors.mean():.5f}'
              )

        g_size.append(subgroup_size)
        g_error.append(subgroup_errors.max())

    plt.scatter(g_size, g_error)
    plt.ylabel('subgroup error')
    plt.xlabel('subgroup size')
    plt.show()
