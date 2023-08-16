
import numpy as np
import pandas as pd

def get_thresholds_ordinal(data_df: pd.DataFrame, min_bin_size, ordinal_size,
                           sigma=0,
                           levels=20):
    """
    Assumes that minimum value is 0
    :param data_df:
    :param min_bin_size:
    :param ordinal_size:
    :return:
    """
    N = len(data_df)
    values = data_df.values
    final_b_edges = [[0], [ordinal_size]]
    for i in range(1, levels):
        num_bins = 2 ** i
        b_edges = np.linspace(0, ordinal_size, num_bins + 1)
        hist = np.histogram(values, bins=b_edges)[0].flatten()
        stats_noised = hist + np.random.normal(0, sigma, size=hist.shape)

        idx = np.argwhere(stats_noised > min_bin_size)
        # edges2 = np.unique(np.concatenate((b_edges[idx+1], b_edges[idx])))
        edges2 = np.unique(np.concatenate((b_edges[idx], b_edges[idx + 1])))
        final_b_edges.append(edges2)
        if num_bins > ordinal_size: break

    edges3 = np.unique(np.concatenate(final_b_edges))
    return edges3


def get_thresholds_realvalued(data_df: pd.DataFrame, min_bin_size: int, levels: int = 20):
    """
    Assumes that minimum value is 0
    :return:
    """
    values = data_df.values.astype(float)
    # final_b_edges = [[-0.0001], [1.001]]
    final_b_edges = []
    for i in range(1, levels):
        num_bins = 2 ** i
        b_edges = np.linspace(0, 1, num_bins + 1)
        b_edges[0] = -0.001
        hist = np.histogram(values, bins=b_edges)[0]
        idx = np.argwhere(hist > min_bin_size).flatten()
        edges2 = np.unique(np.concatenate((b_edges[idx], b_edges[idx + 1])))
        final_b_edges.append(edges2)
    edges3 = np.unique(np.concatenate(final_b_edges))
    return edges3




def get_thresholds_realvalued_v2(data_df: pd.DataFrame, min_bin_size: int, sigma: float = 0, levels: int = 20):
    """
    Assumes that minimum value is 0
    :return:
    """
    values = data_df.values.astype(float)
    # final_b_edges = [[-0.0001], [1.001]]

    thresholds = {}


    final_b_edges = []
    ranges = [(0, 1)]
    for i in range(1, levels):
        level_stats = []
        level_queries = []
        next_ranges = []
        for tup in ranges:
            left, right = tup
            stat = (left < values <= right).mean() + np.random.normal(0, sigma, size=1)
            level_stats.append(stat)
            level_queries.append((left if left > 0 else left -1e-3, right))

            if stat > min_bin_size:
                # Split bin
                mid = (right + left) / 2
                next_ranges.append((left, mid))
                next_ranges.append((mid, right))
            else:
                next_ranges.append((left, right))

        ranges = next_ranges
        thresholds[i] = (level_stats, level_queries)

    edges3 = np.unique(np.concatenate(final_b_edges))
    return edges3
