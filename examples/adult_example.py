import itertools
import sys

sys.path.append("..")
sys.path.append("../..")
from examples.run_example import generate_private_SD, run_experiment
from models import PrivGA, RelaxedProjection
from stats import Marginals, TwoWayPrefix
from utils.utils_data import get_data
import os

if __name__ == "__main__":
    # Get Data
    data = get_data('adult', 'adult', root_path='../data_files/')

    get_gen_list = [
        RelaxedProjection.get_generator(learning_rate=0.01,iterations=3000),
        # PrivGA.get_generator(popsize=10000,
        #                      top_k=top_k,
        #                      num_generations=3000,
        #                      crossover=cross,
        #                      mutations=mut,
        #                      stop_loss_time_window=200,
        #                      print_progress=True)
        # for top_k,cross, mut in itertools.product([50],[0],[1,2,4,8,16])
    ]

    stat_module_list = [
        Marginals.get_all_kway_combinations(data.domain, k=2),
    ]
    result_df = run_experiment(data,
                   data_name='adult',
                   generators=get_gen_list,
                   stat_modules=stat_module_list,
                   epsilon_list=[0.02],
                   # epsilon_list=[1],
                   seed_list=[0],
                   plot_results=True,
                   data_size=100)

    # result_df.to_csv(f'results/top_k_50_parameters_1.csv', index_label=False)