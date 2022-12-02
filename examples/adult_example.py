import sys
# sys.path.append("..")
# sys.path.append("../..")
from examples.run_example import generate_private_SD, run_experiment
from models import PrivGA, RelaxedProjection
from stats import Marginals, TwoWayPrefix

from utils.utils_data import get_data


from folktables import ACSDataSource, ACSIncome
# def run_single_

if __name__ == "__main__":

    # Get Data
    data = get_data('adult', 'adult-mini', root_path='../data_files/')


    get_gen_list = [
        PrivGA.get_generator(popsize=2000,
                             top_k=100,
                             num_generations=400,
                             sigma_scale=0.5,
                             crossover=True,
                             stop_loss_time_window=50,
                             print_progress=True)
    ]

    stat_module_list = [
        Marginals.get_all_kway_combinations(data.domain, k=2),
    ]
    run_experiment(data,
                   data_name='adult',
                   generators=get_gen_list,
                   stat_modules=stat_module_list,
                   epsilon_list=[0.03],
                   # epsilon_list=[0.03, 0.05, 0.07, 0.1, 0.2],
                   seed_list=[0],
                   plot_results=False,
                   data_size=100)


