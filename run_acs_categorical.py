import itertools
import os
import jax
from models import PrivGA, SimpleGAforSyncData, RelaxedProjectionPP
from stats import Halfspace, Prefix, Marginals
import time
from utils.utils_data import get_data
import pandas as pd
import argparse
from run_experiment import run_acs_example


def run_acs_cat(gen_name, queries, task: str, epsilon_list: list, seed_list: list,
                adaptive: bool, rounds_list: list, samples_per_round_list,
                print_progress=False):
    data_name = f'folktables_2018_{task}_CA'
    data = get_data(f'{data_name}-mixed-train',
                    domain_name=f'domain/{data_name}-cat', root_path='data_files/folktables_datasets')

    algos = {
        'PrivGA': PrivGA(num_generations=100000, print_progress=print_progress, strategy=SimpleGAforSyncData(
            domain=data.domain, data_size=2000, population_size=1000, elite_size=2, muta_rate=1, mate_rate=1)),
        'RAP++': RelaxedProjectionPP(domain=data.domain, data_size=1000, learning_rate=(0.0005,),
                                     print_progress=print_progress)
    }
    algo = algos[gen_name]

    bins = [2, 4, 8, 16, 32]
    modules = {
        'Halfspaces': Halfspace.get_kway_random_halfspaces(data.domain, k=1, rng=jax.random.PRNGKey(0), random_hs=150000),
        'Prefix': Prefix.get_kway_prefixes(data.domain, k_cat=1, k_num=2, rng=jax.random.PRNGKey(0), random_prefixes=200000),
        # 'Ranges': Marginals.get_all_kway_mixed_combinations(data.domain, k_disc=1, k_real=1, bins=[2, 4, 8, 16, 32])[0],
        '2-way Marginals': Marginals.get_all_kway_combinations(data.domain, k=2, bins=bins),
        '3-way Marginals': Marginals.get_all_kway_combinations(data.domain, k=3, bins=bins),
    }
    train_module = modules[queries]
    Results = []

    for rounds, samples_per_round, epsilon, seed, in itertools.product(rounds_list, samples_per_round_list,
                                                                       epsilon_list, seed_list):
        max_error, ave_error, total_time = run_acs_example(algo, data,
                                                           data_name=data_name,
                                                           stats_module=train_module,
                                                           epsilon=epsilon,
                                                           seed=seed,
                                                           rounds=rounds,
                                                           num_sample=samples_per_round,
                                                           adaptive=adaptive)
        Results.append(
            [data_name, gen_name, queries, rounds, samples_per_round, epsilon, seed, max_error, ave_error, total_time])

    cols = ['data', 'generator', 'stats', 'T', 'samples', 'epsilon', 'seed', 'max error', 'l1 error', 'time']
    Results_df = pd.DataFrame(Results, columns=cols)
    print(f'Saving ', f'acsreal_{gen_name}_{queries}_results.csv')
    Results_df.to_csv(f'acsreal_{gen_name}_{queries}_results.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ACS Categorical Experiment',
                                     description='Run algorithm PrivGA on categorical ACS data')

    parser.add_argument('--algo', choices=['PrivGA', 'RAP++'], default='PrivGA')
    parser.add_argument('--queries', choices=['3-way Marginals'], default='3-way Marginals')
    parser.add_argument('--task', choices=['employment', 'coverage', 'income', 'mobility', 'travel'],
                        default='income')
    parser.add_argument('--epsilon', type=float, default=[0.07, 0.23, 0.52, 0.74, 1.0], nargs='+')
    parser.add_argument('--seed', type=int, default=[0, 1, 2], nargs='+')
    parser.add_argument('-a', '--adaptive', action='store_true', default=True)  # on/off flag
    parser.add_argument('--rounds', type=int, default=[25, 50, 75, 100], nargs='+')
    parser.add_argument('--samples_per_round', type=int, default=[1], nargs='+')

    args = parser.parse_args()

    run_acs_cat(args.algo, args.queries, args.task, args.epsilon, args.seed, args.adaptive, args.rounds,
                args.samples_per_round, print_progress=True)
