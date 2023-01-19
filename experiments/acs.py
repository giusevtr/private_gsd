import itertools
import time
import pandas as pd
import os
import jax
import jax.random
import jax.numpy as jnp
from models import PrivGAfast, SimpleGAforSyncDataFast, Generator, RelaxedProjection
from stats import Marginals
from utils.utils_data import get_data
from experiments.experiment import run_experiments

"""
Runtime analysis of PrivGA
Split the ACS datasets. 
RAP datasize = 50000
3-way marginals
3 random seeds
T= [25, 50, 100]
"""

def run(EPSILON=(0.07,0.23,0.52,0.74,1.0,)):
    tasks = ['income']
    # states = ['NY', 'CA', 'FL', 'TX', 'PA']
    # tasks = ['coverage']
    states = ['CA']
    RESULTS = []
    print("check1")
    adaptive_rounds = (25,50,75,100,)
    for task, state in itertools.product(tasks, states):
        data_name = f'folktables_2018_{task}_{state}'
        data = get_data(f'folktables_datasets/{data_name}-mixed-train',
                        domain_name=f'folktables_datasets/domain/{data_name}-cat')
        privga = PrivGAfast(
            num_generations=500000,
            print_progress=True,
            strategy=SimpleGAforSyncDataFast(domain=data.domain,
                                             population_size=1000,
                                             elite_size=2,
                                             data_size=2000,
                                             muta_rate=1,
                                             mate_rate=1))
        # rap = RelaxedProjection(domain=data.domain, data_size=1000, iterations=1000, learning_rate=0.05, print_progress=False)
        SEEDS = [2]
        print(SEEDS[0])
        delta = 1.0 / len(data) ** 2
        # stats_module = TwoWayPrefix.get_stat_module(data.domain, num_rand_queries=1000000)
        stats_module,kway_combinations = Marginals.get_all_kway_combinations(data.domain, k=3,bins=(30,))
        # print("workloadaa:",stats_module.get_num_queries())
        stats_module.fit(data)
        run_experiments(data=data, algorithm=privga, stats_module=stats_module, epsilon=EPSILON,
                        adaptive_rounds=adaptive_rounds,
                        seeds=SEEDS,
                        save_dir=('results', data_name, 'PrivGA'),
                        # data_post_processing=lambda data_in: data_in.inverse_map_real_values(col_range)
                        )
    #     for T, eps, seed in itertools.product([100], list(epsilon), [0,1,2]):
    #
    #         for algorithm in ALGORITHMS:
    #             algorithm: Generator
    #             key = jax.random.PRNGKey(seed)
    #             stime = time.time()
    #             sync_data_2 = algorithm.fit_dp_adaptive(key, stat_module=stats_module,
    #                                                     rounds=T, epsilon=eps, delta=delta, print_progress=True, tolerance=0.00,
    #                                                     start_sync=True)
    #             total_time = time.time()-stime
    #             errors = stats_module.get_sync_data_errors(sync_data_2.to_numpy())
    #             max_error = errors.max()
    #             sync_stats = stats_module.private_statistics_fn(sync_data_2)
    #             l1_error = jnp.mean(jnp.abs(stats_module.get_true_statistics() - sync_stats))
    #             RESULTS.append(
    #                 [data_name, str(algorithm), str(stats_module), T, eps, seed, max_error, l1_error, total_time])
    #             print(f'{str(algorithm)}: max error = {errors.max():.5f}')
    #
    #             algo_name = str(algorithm)
    #             save_path = 'sync_datasets'
    #             os.makedirs(save_path, exist_ok=True)
    #             save_path = os.path.join(save_path, data_name)
    #             os.makedirs(save_path, exist_ok=True)
    #             save_path = os.path.join(save_path, algo_name)
    #             os.makedirs(save_path, exist_ok=True)
    #             save_path = os.path.join(save_path, f'{T:03}')
    #             os.makedirs(save_path, exist_ok=True)
    #             save_path = os.path.join(save_path, f'{eps:.2f}')
    #             os.makedirs(save_path, exist_ok=True)
    #             save_path = os.path.join(save_path, f'sync_data_{seed}.csv')
    #             data_df: pd.DataFrame = sync_data_2.df
    #             print(f'Saving {save_path}')
    #             data_df.to_csv(save_path)
    # results_df = pd.DataFrame(RESULTS,
    #                           columns=['data', 'generator', 'stats', 'T', 'epsilon', 'seed', 'max error','l1 error',
    #                                    'time'])
    # results_df.to_csv("result_T_RAP.csv")

if __name__ == "__main__":
    # df = folktables.
    run()
