import itertools
import sys

import pandas as pd
import jax
from models_v2 import PrivGA, RelaxedProjectionPP
from stats_v2 import TwoWayPrefix
from toy_datasets.circles import get_circles_dataset
from visualize.plot_low_dim_data import plot_2d_data
from toy_datasets.sparse import get_sparse_dataset

if __name__ == "__main__":

    # rng = np.random.default_rng()
    # data_np = np.column_stack((rng.uniform(low=0.20, high=0.21, size=(10000, )),
    #                            rng.uniform(low=0.30, high=0.80, size=(10000, ))))

    data = get_sparse_dataset(DATA_SIZE=1000)
    plot_2d_data(data.to_numpy())

    stats_module = TwoWayPrefix.get_stat_module(data.domain, num_rand_queries=1000)
    stats_module.fit(data)

    def plot_circles(X):
        plot_2d_data(X)

    # plot_2d_data(data_np)
    priv_ga = PrivGA(
                     popsize=300,
                    top_k=20,
                    num_generations=500,
                    stop_loss_time_window=50,
                    print_progress=True,
                    start_mutations=32,
                     data_size=200,
                     )
    def debug_fn(X):
        plot_2d_data(X)

    key = jax.random.PRNGKey(0)
    sync_data = priv_ga.fit_dp_adaptive(key, stat_module=stats_module, rounds=30, epsilon=1, delta=1e-6, print_progress=True,
                                        debug_fn=debug_fn)
    errros = stats_module.get_sync_data_errors(sync_data.to_numpy())
    print(f'PrivGA: max error = {errros.max():.5f}')