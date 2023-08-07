
import numpy as np
import pandas as pd

def get_thresholds_ordinal(data_df: pd.DataFrame, min_bin_size, ordinal_size,
                           levels=20):
    """
    Assumes that minimum value is 0
    :param data_df:
    :param min_bin_size:
    :param ordinal_size:
    :return:
    """
    values = data_df.values
    final_b_edges = [[0], [ordinal_size]]
    for i in range(1, levels):
        num_bins = 2 ** i
        b_edges = np.linspace(0, ordinal_size, num_bins + 1)
        hist = np.histogram(values, bins=b_edges)[0]
        idx = np.argwhere(hist > min_bin_size).flatten()
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
