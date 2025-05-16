from snsynth.utils import gaussian_noise, cdp_rho
import numpy as np
import pandas as pd
from genetic_sd.utils.utils import get_sigma


def get_noisy_histogram(values: np.ndarray,  thresholds: np.ndarray, hist_sum: float = 1.0, sigma: float = None):
    """
    Compute the histogram based on thresholds and add noise to provide differential privacy.
    """
    N = values.shape[0]
    stats = np.histogramdd(values, [thresholds])[0].flatten() / N
    if sigma is not None:
        stats = list(np.array(stats) + gaussian_noise(sigma, size=len(stats)))
    stats = np.clip(stats, 0, 1)
    stat_sum = stats.sum()
    if np.abs(stat_sum) < 1e-9:
        stats = hist_sum * np.ones_like(stats) / stats.shape[0]
    else:
        stats = hist_sum * stats / stat_sum
    return stats


def get_thresholds_zcdp_adaptive(data_df: pd.Series,
                                 rho: float = None,
                                 data_range: tuple = (0, 1),
                                 data_granularity: float = 0.01,
                                 num_intervals: int = 32,
                                 verbose=False) -> np.ndarray:
    """
    Compute threshold of the data under differential privacy. The objective is to find num_intervals non-overlapping
    intervals that cover the entire data domain  such that each interval covers the same fraction of points.

    :param data_df: A single column dataset with numeric values
    :param rho: The privacy budget, representing the zCDP parameter.
    :param data_range: the minimum and maximum numeric value of the data.
    :param data_granularity: minumum separation between different data values. For exaple, for integers it's 1
    :param num_intervals:  Number of intervals to represent the values
    :param verbose:
    :return:
    """
    lower, upper = data_range
    range_size = upper - lower
    # How many interval partitions before we achieve an interval that as small as the data granularity
    tree_height = int(np.log2((range_size / num_intervals) / data_granularity) + 1)
    if verbose:
        print(f'tree_height = {tree_height}')

    # The maximum fraction of points that each interval should cover
    min_interval_cover = 1.0 / num_intervals

    N = len(data_df)
    sigma = get_sigma(rho, sensitivity=np.sqrt(2 * (tree_height + 1)) / N) if rho is not None else None
    if verbose:
        print(f'sigma = {sigma}')
    values = data_df.values
    hierarchical_thresholds = {}
    # Begin with uniform intervals
    thresholds = np.linspace(lower, upper, num_intervals+1)
    thresholds[-1] += 1e-6

    # Compute stats for this level based on 'thresholds'
    stats = get_noisy_histogram(values, thresholds, hist_sum=1, sigma=sigma)
    #
    for i in range(0, tree_height):
        level_split = False

        # Split intervals that are bigger than the threshold
        interval_list = []
        for thres_id in range(1, len(thresholds)):
            left = thresholds[thres_id-1]
            right = thresholds[thres_id]
            interval_stat = float(stats[thres_id-1])

            if (interval_stat > min_interval_cover) and ((right - left) > data_granularity ):
                # Split bin if it contains enough information.
                mid = (right + left) / 2

                local_thresholds = np.array([left, mid, right])
                local_stats = get_noisy_histogram(values, local_thresholds, hist_sum=interval_stat, sigma=sigma)

                left_stat = local_stats[0]
                right_stat = local_stats[1]
                interval_list.append((left, mid,   left_stat))
                interval_list.append(( mid, right, right_stat))
                level_split = True
            else:
                interval_list.append((left, right, interval_stat))

        # Extract new intervals:
        new_thresholds = []
        new_stats = []
        for left, right, interval_stat in interval_list:
            new_thresholds.append(left)
            new_stats.append(interval_stat)
        new_thresholds.append(thresholds[-1])

        thresholds = np.array(new_thresholds)
        stats = np.array(new_stats)

        if not level_split:
            # Stop splitting intervals
            total_levels = i
            break

    # Select num_intervals intervals that evenly divide the data based on the private thresholds
    # 1) Sample from the privaate histogram, then compute the num-intervals quantiles
    num_samples = 10000
    num_thresholds = len(thresholds)
    int_ids = np.arange(num_thresholds-1)
    ids = np.random.choice(int_ids, size=num_samples, p=stats)

    left_thres = thresholds[ids]
    right_thres = thresholds[ids+1]
    int_len = right_thres - left_thres
    data = np.random.rand(num_samples) * int_len + left_thres

    final_threshold = np.quantile(data, q=np.linspace(0, 1, num_intervals))

    return final_threshold


def test1():
    import matplotlib.pyplot as plt
    eps = 1.1
    rho = cdp_rho(eps, 1e-6)
    lower, upper = (0, 100123)
    num_intervals = 16
    N = 10000
    values = np.random.randint(lower, upper, N)
    values[:1000] = 10
    values[1000:2000] = 11
    df = pd.Series(values)
    priv_thres = get_thresholds_zcdp_adaptive(df,
                                              rho=rho,
                                    data_range=(lower, upper),
                                    data_granularity=1,
                                    num_intervals=num_intervals,
                                    verbose=False)

    plt.title(f'test_find_thresholds()')
    plt.hist(values, bins=priv_thres)
    for t in priv_thres:
        plt.vlines(x=t, ymin=0, ymax=2500, colors='k', alpha=0.1)
    plt.xscale('log')
    plt.show()
    print()


if __name__ == "__main__":
    test1()
