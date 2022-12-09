import itertools
import sys
# sys.path.append("..")
# sys.path.append("../..")
from examples.run_example import generate_private_SD, run_experiment
from models import PrivGA, RelaxedProjectionPP
from stats import LogRegQuery
from toy_datasets.circles import get_circles_dataset
from toy_datasets.classification import get_classification
from visualize.plot_low_dim_data import plot_2d_data
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    DIM = 2
    data = get_classification(DATA_SIZE=1000, d=DIM)
    get_gen_list = [
        # RelaxedProjectionPP.get_generator(learning_rate=(0.001, 0.005, 0.01)),
        PrivGA.get_generator(popsize=1000,
                            top_k=30,
                            num_generations=20,
                             crossover=cross,
                             mutations=mut,
                            stop_loss_time_window=50,
                            print_progress=True)
        # for cross, mut in itertools.product([1, 2, 4, 8], [1, 2, 4, 8])
        for cross, mut in itertools.product([1], [1])
    ]

    stat_module_list = [
        LogRegQuery(data.domain, DIM),
        # Marginals.get_all_kway_combinations(data_small.domain, k=3),
    ]

    def plot_data(X):
        plt.figure(figsize=(5, 5))
        # plt.title(title)
        plt.scatter(X[:, 0], X[:, 1], c=X[:, 2])
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()

    plot_data(data.to_numpy())
    run_experiment(data,
                   data_name='class-2d',
                   generators=get_gen_list,
                   stat_modules=stat_module_list,
                   epsilon_list=[1000],
                   seed_list=[0],
                   data_size=50,
                   plot_results=False,
                   callback_fn=plot_data)
