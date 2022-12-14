import itertools
import sys

import pandas as pd

# sys.path.append("..")
# sys.path.append("../..")
from examples.run_example import generate_private_SD, run_experiment
from models import PrivGA, RelaxedProjectionPP
from stats import TwoWayPrefix
from toy_datasets.circles import get_circles_dataset
from visualize.plot_low_dim_data import plot_2d_data
import numpy as np
from utils import Dataset, Domain
if __name__ == "__main__":

    rng = np.random.default_rng()
    data_np = np.column_stack((rng.uniform(low=0.20, high=0.21, size=(10000, )),
                               rng.uniform(low=0.30, high=0.80, size=(10000, ))))

    cols = ['A', 'B']
    shape = [1, 1]
    domain = Domain(cols, shape)
    data_df = pd.DataFrame(data_np, columns=cols)
    data = Dataset(data_df, domain)

    get_gen_list = [
        # RelaxedProjectionPP.get_generator(learning_rate=(0.001, 0.005, 0.01)),
        PrivGA.get_generator(popsize=100,
                            top_k=5,
                            num_generations=1000,
                            stop_loss_time_window=50,
                             start_mutations=30,
                            print_progress=True)
    ]

    stat_module_list = [
        TwoWayPrefix(data.domain, num_rand_queries=1000),
        # Marginals.get_all_kway_combinations(data_small.domain, k=3),
    ]

    def plot_circles(X):
        plot_2d_data(X)

    plot_2d_data(data_np)
    run_experiment(data,
                   data_name='sparse-2d',
                   generators=get_gen_list,
                   stat_modules=stat_module_list,
                   epsilon_list=[100],
                   seed_list=[0],
                   data_size=500,
                   plot_results=False,
                   callback_fn=plot_circles)
