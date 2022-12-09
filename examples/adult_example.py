import itertools
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append("..")
sys.path.append("../..")
from examples.run_example import generate_private_SD, run_experiment
from models import PrivGA, RelaxedProjection
from stats import Marginals, TwoWayPrefix
from utils.utils_data import get_data
import pdb


if __name__ == "__main__":
    # Get Data

    data = get_data('adult', 'adult', root_path='../data_files/')

    get_gen_list = [
        RelaxedProjection.get_generator(learning_rate=lr,iterations=10000)
        # PrivGA.get_generator(popsize=100,
        #                      top_k=2,
        #                      num_generations=20000,
        #                      crossover=0,
        #                      mutations=2,
        #                      stop_loss_time_window=500,
        #                      print_progress=True)
        for lr in [0.0001,0.0005,0.001,0.002,0.005,0.1,0.2,0.5,1]
    ]

    stat_module_list = [
        Marginals.get_all_kway_combinations(data.domain, k=2),
    ]
    result_df = run_experiment(data,
                   data_name='adult',
                   generators=get_gen_list,
                   stat_modules=stat_module_list,
                   epsilon_list=[0.02,0.2],
                   # epsilon_list=[0.07, 0.15, 0.23, 0.41, 0.52, 0.62, 0.74, 0.87, 1.0],
                   seed_list=[0,1,2,3,4],
                   plot_results=True,
                   data_size=1000)

    result_df.to_csv(f'results/rp_parameters.csv')