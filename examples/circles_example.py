import sys
# sys.path.append("..")
# sys.path.append("../..")
from examples.run_example import generate_private_SD, run_experiment
from models import PrivGA, RelaxedProjectionPP
from stats import TwoWayPrefix
from toy_datasets.circles import get_circles_dataset
from visualize.plot_low_dim_data import plot_2d_data

if __name__ == "__main__":

    data = get_circles_dataset()
    get_gen_list = [
        # RelaxedProjectionPP.get_generator(learning_rate=(0.001, 0.005, 0.01)),
        PrivGA.get_generator(popsize=5000,
                            top_k=100,
                            num_generations=150,
                            sigma_scale=0.3,
                             crossover=False,
                            stop_loss_time_window=50,
                            print_progress=True),

        # PrivGA.get_generator(popsize=5000,
        #                      top_k=100,
        #                      num_generations=150,
        #                      sigma_scale=0.5,
        #                      crossover=False,
        #                      stop_loss_time_window=50,
        #                      print_progress=True)
    ]

    stat_module_list = [
        TwoWayPrefix(data.domain, num_rand_queries=1000),
        # Marginals.get_all_kway_combinations(data_small.domain, k=3),
    ]

    def plot_circles(X):
        plot_2d_data(X)

    run_experiment(data,
                   data_name='circles-2d',
                   generators=get_gen_list,
                   stat_modules=stat_module_list,
                   epsilon_list=[100],
                   seed_list=[0],
                   data_size=100,
                   plot_results=False,
                   callback_fn=plot_circles)
