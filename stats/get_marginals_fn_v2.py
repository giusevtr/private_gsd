try:
    from snsynth.utils import gaussian_noise, cdp_rho
except:
    from snsynth.utils import gaussian_noise, cdp_rho

from jax.config import config; config.update("jax_enable_x64", True)
import numpy as np
import jax
import jax.numpy as jnp
import chex
import pandas as pd
from utils import Dataset, Domain
import itertools
def get_sigma(rho: float,  sensitivity: float) -> float:
    if rho is None:
        return 0.0
    return np.sqrt(sensitivity**2 / rho)

def _divide_privacy_budget(rho: float, t: int) -> float:
    if rho is None: return None
    return rho / t

def get_thresholds_categorical(data_df: pd.Series,
                           size,
                           rho: float = None):
    """
    TODO: Add a maximum size parameter for extremely large cardinality features
    """
    N = len(data_df)
    sigma = get_sigma(rho, sensitivity=np.sqrt(2) / N)
    values = data_df.values

    bin_edges = np.linspace(0, size, size + 1) - 0.001
    hist, b_edges = np.histogramdd(values, bins=[bin_edges])
    stats = hist.flatten() / N
    stats_noised = stats + gaussian_noise(sigma=sigma, size=hist.shape[0])

    thresholds = {0: {'stats': stats_noised, 'bins': bin_edges}, 'levels' : 1}
    return thresholds


def get_thresholds_ordinal(data_df: pd.Series,
                           min_bin_size, ordinal_size,
                           rho: float = None,
                           verbose=False
                           ) -> dict:
    levels = int(np.log2(ordinal_size)+1)
    res: dict
    res = get_thresholds_realvalued(data_df, min_bin_size, levels, rho, range_size=ordinal_size, verbose=verbose)
    return res



def get_thresholds_realvalued(data_df: pd.Series, min_bin_size: float, levels: int = 20, rho: float = None,
                              range_size=1, verbose=False) -> (dict, list):
    """
    Assumes that minimum value is 0
    """
    N = len(data_df)
    sigma = get_sigma(rho, sensitivity=np.sqrt(2 * levels) / N)
    values: np.array
    values = data_df.values
    hierarchical_thresholds = {}
    thresholds = [0, range_size/2, range_size]
    total_levels = levels
    for i in range(0, levels):
        level_split = False
        thresholds.sort()

        # Compute stats for this level based on 'thresholds'
        thresholds_temp = thresholds.copy()
        thresholds_temp[-1] += 1e-6
        stats = np.histogramdd(values, [thresholds_temp])[0].flatten() / N
        stats_priv = list(np.array(stats) + gaussian_noise(sigma, size=len(stats)))

        # Record stats and thresholds
        hierarchical_thresholds[i] = {'stats': stats_priv, 'bins': np.array(thresholds_temp)}

        for thres_id in range(1, len(thresholds)):
            left = thresholds[thres_id-1]
            right = thresholds[thres_id]
            interval_stat = stats_priv[thres_id-1]

            if (i <= 4) or (interval_stat > min_bin_size + 2*sigma):
                # Split bin if it contains enough information. Always split levels<=4.
                mid = (right + left) / 2
                thresholds.append(mid)
                level_split = True

        if not level_split:
            # Stop splitting intervals
            total_levels = i
            break

    hierarchical_thresholds['levels'] = total_levels
    return hierarchical_thresholds, thresholds

# def get_thresholds_realvalued(data_df: pd.Series, min_bin_size: float, levels: int = 20, rho: float = None,
#                               range_size=1, verbose=False) -> (dict, list):
#     """
#     Assumes that minimum value is 0
#     """
#     N = len(data_df)
#     sigma = get_sigma(rho, sensitivity=np.sqrt(2 * levels) / N)
#     values: np.array
#     values = data_df.values
#     thresholds = []
#     hierarchical_thresholds = {}
#     intervals = [(0, range_size/2), (range_size/2, range_size)]
#     total_levels = levels
#     for i in range(0, levels):
#         level_stats = []
#         level_intervals = []
#         next_ranges = []
#         level_split = False
#         for tup in intervals:
#             left, right = tup
#             # left_temp = left if left > 0 else left -1e-3  # zero is a special case
#             right_temp = right if right < 1 else right + 1e-8  # zero is a special case
#             stat = ((left <= values) & (values < right_temp)).astype(int).mean() + gaussian_noise(sigma, size=1)
#             level_stats.append(stat[0])
#             level_intervals.append(right_temp)
#             thresholds.append(right)
#             if (i <= 4) or (stat > min_bin_size + 2*sigma):
#                 # Split bin if it contains enough information. Always split levels<=4.
#                 mid = (right + left) / 2
#                 next_ranges.append((left, mid))
#                 next_ranges.append((mid, right))
#                 level_split = True
#             else:
#                 next_ranges.append((left, right))
#
#         intervals = next_ranges
#         level_intervals = np.array([0.0] + level_intervals, dtype=float)
#         hierarchical_thresholds[i] = {'stats': level_stats, 'bins': level_intervals}
#         if not level_split:
#             total_levels = i
#             break
#     thresholds.append(0)
#     hierarchical_thresholds['levels'] = total_levels
#     thresholds = list(set(thresholds))
#     thresholds.sort()
#     return hierarchical_thresholds, thresholds





def _get_stats_fn(k, query_params):
    these_queries = jnp.array(query_params, dtype=jnp.float64)

    def answer_fn(x_row: chex.Array, query_single: chex.Array):
        I = query_single[:k].astype(int)
        U = query_single[k:2 * k]
        L = query_single[2 * k:3 * k]
        t1 = (x_row[I] < U).astype(int)
        t2 = (x_row[I] >= L).astype(int)
        t3 = jnp.prod(jnp.array([t1, t2]), axis=0)
        answers = jnp.prod(t3)
        return answers

    temp_stat_fn = jax.vmap(answer_fn, in_axes=(None, 0))

    def scan_fun(carry, x):
        return carry + temp_stat_fn(x, these_queries), None

    def stat_fn(X):
        out = jax.eval_shape(temp_stat_fn, X[0], these_queries)
        stats = jax.lax.scan(scan_fun, jnp.zeros(out.shape, out.dtype), X)[0]
        stats = jnp.round(stats)
        return stats / X.shape[0]

    return stat_fn


def _get_query_params(data, indices: list, bin_edges) -> (list, list):
    # indices = [data.domain.get_attribute_index(feature) for feature in features]
    answer_vectors = []
    query_params = []
    values = data.to_numpy_np()
    N = values.shape[0]
    bin_indices = [np.arange(bins.shape[0])[1:] for bins in bin_edges]
    bin_positions = list(itertools.product(*bin_indices))
    hist, b_edges = np.histogramdd(values[:, indices], bins=bin_edges)
    stats = hist.flatten() / N
    k = len(bin_edges)
    for stat_id in np.arange(stats.shape[0]):
        bins_idx: tuple
        bins_idx = bin_positions[stat_id]
        lower = []
        upper = []
        for i in range(k):
            bin_id: int
            bin_id = bins_idx[i]
            lower_threshold = bin_edges[i][bin_id - 1]
            upper_threshold = bin_edges[i][bin_id]
            lower.append(lower_threshold)
            upper.append(upper_threshold)
        query_params.append(np.concatenate((np.array(indices).astype(int), np.array(upper), np.array(lower))))
        answer_vectors.append(stats[stat_id])
    return answer_vectors, query_params

####################################################################################################
def _get_thresholds_1way_marginal_fn(data: Dataset,
                                     min_bin_density: float = 0.01,
                                     levels=20,
                                     rho: float = None,
                                     verbose=False
                                     ):
    """
    This function will compute the 1-way marginals functin and
    output the bin_edges of all features.
    """
    num_columns = len(data.domain.attrs)
    rho_div = _divide_privacy_budget(rho, num_columns)
    domain = data.domain
    query_params = []
    total_stats = []
    N = len(data.df)
    cat_features = domain.get_categorical_cols()
    num_features = domain.get_numerical_cols()
    ord_features = domain.get_ordinal_cols()

    # Compute bin_edges
    bins = {}
    quantiles = {}
    # min_bin_size = int(N * (min_bin_density / 2) + 1)  # Number of points on each edge
    for col_name in domain.get_numerical_cols():
        bins[col_name], quantiles[col_name] = get_thresholds_realvalued(data.df[col_name], min_bin_density, levels=levels, rho=rho_div, verbose=verbose)
        if verbose: print(f'{col_name}: levels=', bins[col_name]['levels'], 'quantiles=', len(quantiles[col_name]))
    for col_name in domain.get_ordinal_cols():
        bins[col_name], quantiles[col_name] = get_thresholds_ordinal(data.df[col_name], min_bin_density, domain.size(col_name), rho=rho_div, verbose=verbose)
        if verbose: print(f'{col_name}: levels=', bins[col_name]['levels'], 'quantiles=', len(quantiles[col_name]))
    for col_name in domain.get_categorical_cols():
        bins[col_name] = get_thresholds_categorical(data.df[col_name], domain.size(col_name), rho=rho_div)


    for feat in cat_features:
        feat_bins: dict
        feat_bins = bins[feat]
        indices = [domain.get_attribute_index(feat)]
        bin_edges = [feat_bins[0]['bins']]
        ans, q_params = _get_query_params(data, indices, bin_edges)
        total_stats += list(feat_bins[0]['stats'])
        query_params += q_params

    for real_feat in num_features:
        indices = [domain.get_attribute_index(real_feat)]
        feat_levels = bins[real_feat]['levels']
        for level in range(feat_levels):
            bin_edges = [bins[real_feat][level]['bins']]
            priv_stats = bins[real_feat][level]['stats']
            ans, q_params = _get_query_params(data, indices, bin_edges)
            total_stats += list(priv_stats)
            query_params += q_params

    for ord_feat in ord_features:
        indices = [domain.get_attribute_index(ord_feat)]
        ord_levels = bins[ord_feat]['levels']
        for level in range(ord_levels):
            bin_edges = [bins[ord_feat][level]['bins']]
            priv_stats = bins[ord_feat][level]['stats']
            ans, q_params = _get_query_params(data, indices, bin_edges)
            total_stats += list(priv_stats)
            query_params += q_params
    return np.array(total_stats), _get_stats_fn(k=1, query_params=query_params), bins, quantiles

def _get_truncated_stats(stats_noised: list, query_params: list, min_density: float=None, max_size: int = None) -> (list, list, float):
    # returns the most informative queries.
    stats_noised_np = np.array(stats_noised)
    stats_ids_sorted = np.argsort(-stats_noised_np)
    if max_size is not None and stats_ids_sorted.shape[0] > max_size:
        stats_ids_sorted = stats_ids_sorted[:max_size]
    density_sum = 0
    truc_stats_noised = []
    truc_query_params = []

    for stat_id in stats_ids_sorted:
        stat_temp = stats_noised[stat_id]
        density_sum += stat_temp
        truc_stats_noised.append(stat_temp)
        truc_query_params.append(query_params[stat_id])
        if min_density is not None and density_sum > min_density: break
    return truc_stats_noised, truc_query_params, density_sum


def _get_bins_level(bin_edges: dict, level):
    max_level = bin_edges['levels'] - 1
    return bin_edges[min(max_level, level)]['bins']


def _get_mixed_marginal_fn(data: Dataset,
                           k_real: int,
                           bin_edges: dict,
                           maximum_size: int = None,
                           conditional_column: list=(),
                           rho: float = None,
                           trucate_density: float = None,
                           output_query_params: bool = False,
                           verbose=False):
    domain = data.domain

    features = domain.get_numerical_cols() + domain.get_ordinal_cols() + domain.get_categorical_cols()
    for c in conditional_column:
        features.remove(c)
    k_total = k_real + len(conditional_column)
    marginals_list = []
    for marginal in [list(idx) for idx in itertools.combinations(features, k_real)]:
        marginals_list.append(marginal)
    total_marginals = len(marginals_list)

    # Divide privacy budget and query capacity
    N = len(data.df)
    rho_split = _divide_privacy_budget(rho, total_marginals)
    query_params_total = []
    private_stats_total = []
    for marginal in marginals_list:
        cond_marginal = list(marginal) + list(conditional_column)

        num_col_levels = [bin_edges[col]['levels'] for col in marginal]
        top_level = max(num_col_levels)
        marginal_max_size = maximum_size // (total_marginals) if maximum_size else None
        # marginal_max_size = min(marginal_max_size, N) if marginal_max_size else None
        sigma = get_sigma(rho_split, sensitivity=np.sqrt(2 * top_level) / N)
        if verbose:
            print('Cond.Marginal=', cond_marginal, f'. Sigma={sigma:.4f}. Top.Level={top_level}. Max.Size={marginal_max_size}')

        # Get the answers and select the bins with most information
        # indices = domain.get_attribute_indices(marginal)
        indices = [domain.get_attribute_index(col_name) for col_name in cond_marginal]

        marginal_stats = None
        marginal_params = None
        density = None
        L = None
        for L in range(top_level):
            kway_edges = [_get_bins_level(bin_edges[col_name], L) for col_name in marginal] + \
                         [bin_edges[cat_col][0]['bins'] for cat_col in conditional_column]
            stats, query_params = _get_query_params(data, indices, kway_edges)
            priv_stats = np.array(stats)
            if sigma > 0: priv_stats =  priv_stats + gaussian_noise(sigma=sigma, size=len(stats)) # Add DP

            priv_stats_truc, query_params_truc, density_temp = _get_truncated_stats(priv_stats, query_params,
                                                                                    min_density=trucate_density,
                                                                                    max_size=marginal_max_size)
            if density_temp < 0.999: break
            density = density_temp
            marginal_stats = priv_stats_truc
            marginal_params = query_params_truc

        private_stats_total += marginal_stats
        query_params_total += marginal_params
        if verbose:
            print(f'\tMarginal size = {len(marginal_stats):<5}.Density={density:.5f}. Level={L-1}')

    if verbose:
        print(f'\tTotal size={len(private_stats_total)}')

    if output_query_params:
        return private_stats_total, _get_stats_fn(k_total, query_params_total), k_total, query_params_total
    return private_stats_total, _get_stats_fn(k_total, query_params_total)


##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################

def get_private_marginal_fn(data: Dataset,
                            eps, delta,
                            maximum_queries: int = None,
                            conditional_features = (),
                            verbose=False
                       ):
    rho = cdp_rho(eps, delta)
    domain = data.domain

    stats_list = []
    stat_fn_list = []

    stats_1_way, stat_1_way_fn, bins, quantiles = _get_thresholds_1way_marginal_fn(data, rho=rho, verbose=True)
    if domain.get_numerical_cols() > 2:
        stats_2way_real, stat_2way_real_fn = _get_mixed_marginal_fn(data, k_real=2, bin_edges=bins,
                                                                    maximum_size=maximum_queries,
                                                                    conditional_column=conditional_features,
                                                                    rho=rho,
                                                                    verbose=verbose)
        stats_list.append(stats_2way_real)
        stat_fn_list.append(stat_2way_real_fn)





def get_marginal(data: Dataset, conditional_col: list = (),
                min_bin_density=1/100,
                 levels=20,
                 maximum_size=1000000, verbose=False):
    """ Return the marginal function without privacy """
    stats_1_way, stat_1_way_fn, bins, quantiles = _get_thresholds_1way_marginal_fn(data,
                                                                                   min_bin_density=min_bin_density
                                                                                   , levels=levels, verbose=verbose)
    stats_k_way, stat_k_way_fn = _get_mixed_marginal_fn(data, k_real=2, bin_edges=bins,
                                              maximum_size=maximum_size,
                                                trucate_density=0.999,
                                              conditional_column=conditional_col, verbose=verbose)

    data.domain.set_bin_edges(quantiles)
    # def stat_fn(X):
    #     s1 = stat_1_way_fn(X)
    #     sk = stat_k_way_fn(X)
    #     return jnp.concatenate([s1, sk])
    # true_stats = jnp.concatenate([jnp.array(stats_1_way), jnp.array(stats_k_way)])
    # return true_stats, stat_fn
    def stat_1way_fn(X):
        s1 = stat_1_way_fn(X)
        return s1
    true_stats_1way = jnp.array(stats_1_way)

    def stat_fn(X):
        s1 = stat_k_way_fn(X)
        return s1
    true_stats = jnp.array(stats_k_way)
    return true_stats, stat_fn


class Query:
    def __init__(self, data: Dataset, conditional_col: list = (),
                min_bin_density=1/100,
                 levels=20,
                 maximum_size=1000000, verbose=False):
        self.K = 2
        stats_1_way, stat_1_way_fn, bins, self.quantiles = _get_thresholds_1way_marginal_fn(data,
                                                                                       min_bin_density=min_bin_density
                                                                                       , levels=levels, verbose=verbose)
        stats_k_way, self.stat_k_way_fn, self.K, self.query_params = _get_mixed_marginal_fn(data, k_real=self.K, bin_edges=bins,
                                                            maximum_size=maximum_size,
                                                            trucate_density=0.999,
                                                            conditional_column=conditional_col,
                                                            output_query_params=True,
                                                            verbose=verbose)
        data.domain.set_bin_edges(self.quantiles)
        self.true_stats = jnp.array(stats_k_way)


    def get_true_stats(self):
        return self.true_stats

    def get_all_stats(self, arg_data: Dataset):
        return self.stat_k_way_fn(arg_data.to_numpy())

    def get_stats_fn(self, query_ids):
        assert len(query_ids) > 0
        sub_query_params = []
        for i in query_ids:
            sub_query_params.append(self.query_params[i])
        sub_stat_fn = _get_stats_fn(self.K, query_params=sub_query_params)
        return sub_stat_fn
