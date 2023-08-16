import numpy as np
import jax
import jax.numpy as jnp
import chex
import pandas as pd
from utils import Dataset, Domain
import itertools
from stats.thresholds import get_thresholds_ordinal, get_thresholds_realvalued


def get_marginal_query(seed, data: Dataset, domain: Domain,
                       k: int = 2,
                       min_bin_density: float = 0.0,
                       minimum_density=1,
                       max_marginal_size: int = None,
                       min_marginal_size: int = 1000,
                       sigma: float = 0,
                       include_features=(),
                       verbose=False):
    """
        The data object represents a normalize dataset, where each real valued feature is bounded in [0,1]
    and both categorical and ordinal features are integers [0, ..., sz].
        This function computes the marginal queries that contain the most information about data. The parameters
    marginal_density and marginal_size control the size of each marginal. It also returns the answers.
        If the privacy parameter rho is passed, then it computes the queries and answers using zCDP.

    output statistics, statistics_fn:
    - [0, 1] only include statistical queries with answers > min_bin_density
    """
    rng = np.random.default_rng(seed)

    N = len(data.df)
    values = data.to_numpy_np()

    # 1) Compute 1-way marginals and bins of each feature.
    features_bin_edges = {}
    features_bin_indices = {}
    min_bin_size = int(N * (min_bin_density / 2) + 1)  # Number of points on each edge
    for col_name in domain.get_numerical_cols():
        features_bin_edges[col_name] = get_thresholds_realvalued(data.df[col_name], min_bin_size, levels=20)
    for col_name in domain.get_ordinal_cols():
        features_bin_edges[col_name] = get_thresholds_ordinal(data.df[col_name], min_bin_size, domain.size(col_name), levels=20)-0.001
    for col_name in domain.get_categorical_cols():
        features_bin_edges[col_name] = np.linspace(0, domain.size(col_name), domain.size(col_name)+1)-0.001

    domain.set_bin_edges(features_bin_edges)
    for col_name in domain.attrs:
        features_bin_indices[col_name] = np.arange(features_bin_edges[col_name].shape[0])
        if verbose: print(f'{col_name} has {features_bin_indices[col_name].shape[0]} thresholds.')


    # 2) Compute k-way marginals using
    query_params = []
    total_stats = []
    stats_sum = 0

    k_temp = k
    k = k + len(include_features)
    features = [f for f in domain.attrs if f not in list(include_features)]

    # Use this for debuging
    def total_error_fn(data_real: Dataset, data_sync: Dataset):
        data_real_np = data_real.to_numpy_np()
        data_sync_np = data_sync.to_numpy_np()
        marginal_results = []
        for marginal in [list(idx) for idx in itertools.combinations(features, k_temp)]:
            marginal = marginal + list(include_features)
            indices = [domain.get_attribute_indices([col_name])[0] for col_name in marginal]
            bin_edges = [features_bin_edges[col_name] for col_name in marginal]
            hist_real, _ = np.histogramdd(data_real_np[:, indices], bins=bin_edges)
            hist_sync, _ = np.histogramdd(data_sync_np[:, indices], bins=bin_edges)
            stats_real = hist_real.flatten() / data_real_np.shape[0]
            stats_sync = hist_sync.flatten() / data_sync_np.shape[0]
            error_avg = np.linalg.norm(stats_real - stats_sync, ord=1) / stats_sync.shape[0]
            error_sq = np.linalg.norm(stats_real - stats_sync, ord=2) / stats_sync.shape[0]
            error_max = np.abs(stats_real - stats_sync).max()
            mar_res = [str(marginal), error_max, error_avg, error_sq]
            marginal_results.append(mar_res)
        marginal_results_df = pd.DataFrame(marginal_results, columns=['Marginal', 'Max', 'Average', 'Avg Squared'])
        return marginal_results_df

    for marginal in [list(idx) for idx in itertools.combinations(features, k_temp)]:
        marginal = marginal + list(include_features)

        # Get the answers and select the bins with most information
        # indices = domain.get_attribute_indices(marginal)
        indices = [domain.get_attribute_indices([col_name])[0] for col_name in marginal]
        bin_edges = [features_bin_edges[col_name] for col_name in marginal]
        hist, b_edges = np.histogramdd(values[:, indices], bins=bin_edges)
        stats = hist.flatten() / N
        stats_noised = stats + np.random.normal(0, sigma, size=stats.shape)
        stats_sum += stats_noised.shape[0]
        bin_positions = list(itertools.product(*[features_bin_indices[col_name][1:] for col_name in marginal]))

        stats_ids_sorted = np.argsort(-stats_noised)
        if stats_ids_sorted.shape[0] > max_marginal_size:
            stats_ids_sorted = stats_ids_sorted[:max_marginal_size]
        # for bins_idx in bin_positions:
        informative_stats_ids = []
        density_sum = 0
        for stat_id in stats_ids_sorted:
            stat_temp = stats_noised[stat_id]
            density_sum += stat_temp

            bins_idx = bin_positions[stat_id]
            lower = []
            upper = []
            for i in range(k):
                bin_id = bins_idx[i]
                lower_threshold = features_bin_edges[marginal[i]][bin_id-1]
                upper_threshold = features_bin_edges[marginal[i]][bin_id]
                lower.append(lower_threshold)
                upper.append(upper_threshold)
            query_params.append(np.concatenate((np.array(indices), np.array(upper), np.array(lower))))
            informative_stats_ids.append(stat_id)

            if density_sum >= 1.0-1e-8:
                break
            if density_sum >= minimum_density-1e-8:
                if min_marginal_size is not None and len(informative_stats_ids) >= min_marginal_size:
                    break

        # Add stats to answers in the same order of the queries.
        final_stats = stats_noised[np.array(informative_stats_ids)]
        density_left = (1 - density_sum) / final_stats.shape[0]
        final_stats2 = final_stats + density_left  # inflate statistics
        total_stats.append(final_stats2)

        if verbose:
            print(f'Marginal {marginal}: Final queries is {final_stats.shape[0]:<7} of {stats.shape[0]:<7}.'
                  f'Density = {final_stats.sum():.8f}. Density2 = {final_stats2.sum():.8f}')
    these_queries = jnp.array(query_params)

    def answer_fn(x_row: chex.Array, query_single: chex.Array):
        I = query_single[:k].astype(int)
        U = query_single[k:2 * k]
        L = query_single[2 * k:3 * k]
        t1 = (x_row[I] <= U).astype(int)
        t2 = (x_row[I] > L).astype(int)
        t3 = jnp.prod(jnp.array([t1, t2]), axis=0)
        answers = jnp.prod(t3)
        return answers

    temp_stat_fn = jax.vmap(answer_fn, in_axes=(None, 0))

    def scan_fun(carry, x):
        return carry + temp_stat_fn(x, these_queries), None
    def stat_fn(X):
        out = jax.eval_shape(temp_stat_fn, X[0], these_queries)
        stats = jax.lax.scan(scan_fun, jnp.zeros(out.shape, out.dtype), X)[0]
        return stats / X.shape[0]

    total_stats_arr = jnp.concatenate(total_stats)
    if verbose:
        print(f'Using {total_stats_arr.shape[0]} out of {stats_sum}.')
    return total_stats_arr, stat_fn, total_error_fn


if __name__ == "__main__":
    domain = Domain({
                     # 'A':{'type':'categorical', 'size': 1},
                     # 'B':{'type':'categorical', 'size': 50},
                     # 'C':{'type':'categorical', 'size': 50},
                     # 'D':{'type':'categorical', 'size': 50},
                     # 'E':{'type':'ordinal', 'size': 10},
                     'F':{'type':'numerical', 'size': 1},
                     })
    data = Dataset.synthetic(domain, 10, 0)
    # data.df.loc[:5000-1, ['A']] = np.zeros((5000,1))
    # data.df.loc[5000:, ['A']] = np.ones((5000, 1))
    # data.df.loc[:1000-1, ['B']] = np.ones((1000, 1))
    # data.df.loc[:, ['F']] = np.ones((10, 1))
    data.df.loc[:, ['F']] = np.zeros((10, 1))
    # data.df.loc[5:, ['F']] = 0.00000190734863 * np.ones((5, 1))

    stats0, marginal_fn = get_marginal_query(0, data, domain, 1)

    stats = marginal_fn(data.to_numpy())

    print(stats0)
    print(stats)
    print(jnp.linalg.norm(stats - stats0))

