from experiments.param_search import  param_search
import itertools
import jax
from models import Generator, PrivGAfast, SimpleGAforSyncDataFast
from stats import Marginals
from toy_datasets.sparse import get_sparse_dataset
import matplotlib.pyplot as plt
import time
# from plot import plot_sparse


if __name__ == "__main__":
    data = get_sparse_dataset(DATA_SIZE=10000)

    bins = [2, 4, 8, 16, 32, 64]
    stats_module, kway_combinations = Marginals.get_all_kway_mixed_combinations(data.domain, k_disc=1, k_real=2,
                                                                                bins=bins)
    stats_module.fit(data)
    # Get Data

    param_search(data, 'sparse', stats_module)