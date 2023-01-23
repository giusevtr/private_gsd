import itertools
import os
import jax
import pandas as pd

from models import PrivGAfast, SimpleGAforSyncDataFast, RelaxedProjectionPP
from stats import Halfspace, Prefix, Marginals
import time
from utils.utils_data import get_data
from utils import Dataset
import argparse


def run_acs_example(algo,
                    data,
                    stats_module,
                    epsilon=1.00,
                    seed=0,
                    rounds=30,
                    num_sample=10,
                    adaptive=False):



    print(f'Running {algo} with epsilon={epsilon}, train module is {stats_module}')
    if adaptive:
        print(f'Adaptive with {rounds} rounds and {num_sample} samples.')
    else:
        print('Non-adaptive')
    stats_module.fit(data)
    folder = f'sync_data/{str(algo)}/{stats_module}/{rounds:03}/{epsilon:.2f}'
    os.makedirs(folder, exist_ok=True)
    path = f'{folder}/sync_data_{seed}.csv'

    print(f'Starting {algo}:')
    stime = time.time()
    key = jax.random.PRNGKey(seed)

    print(f'Saving in {path}')
    sync_data.df.to_csv(path, index=False)

    errors = jax.numpy.abs(stats_module.get_true_stats() - stats_module.get_stats_jit(sync_data))
    ave_error = jax.numpy.linalg.norm(errors, ord=1)/errors.shape[0]
    print(f'{str(algo)}: Train max error = {errors.max():.4f}, ave_error={ave_error:.6f}, time={time.time()-stime:.4f}')



def run_all_acs(generators,  queries, epsilon: list, seed: list,  rounds:list):

    Results = []
    data_name = f'folktables_2018_real_CA'
    data = get_data(f'{data_name}-mixed-train',
                    domain_name=f'domain/{data_name}-mixed', root_path='data_files/folktables_datasets_real')
    modules = {
            'Halfspaces': Halfspace.get_kway_random_halfspaces(data.domain, k=1, rng=jax.random.PRNGKey(0), random_hs=20000)[0],
            'Prefix': Prefix.get_kway_prefixes(data.domain, k=1, rng=jax.random.PRNGKey(0), random_prefixes=20000)[0],
            'Ranges': Marginals.get_all_kway_mixed_combinations(data.domain, k_disc=1, k_real=2, bins=[2, 4, 8, 16, 32, 64])[0]
        }
    for query_set in queries:

        train_module = modules[query_set]
        print(f'fitting data on module {query_set}...', end=' ')
        train_module.fit(data)
        print(f'Done fitting.')

        for gen_name, rounds, epsilon, seed in itertools.product(generators,  rounds, epsilon, seed):

            sync_path = f'sync_data/{gen_name}/{query_set}/{rounds:03}/{epsilon:.2f}/sync_data_{seed}.csv'
            if not os.path.exists(sync_path):
                print(f'\tCannot find {sync_path}')
                continue
            print(f'reading {sync_path}')
            df = pd.read_csv(sync_path, index_col=None)
            sync_data = Dataset(df, data.domain)

            errors = jax.numpy.abs(train_module.get_true_stats() - train_module.get_stats_jit(sync_data))
            max_error = errors.max()
            ave_error = jax.numpy.linalg.norm(errors, ord=1)/errors.shape[0]
            print(f'{gen_name}, {query_set}, {rounds}, {epsilon}: Train max error = {errors.max():.4f}, ave_error={ave_error:.6f}')

            Results.append([data_name, gen_name, query_set, rounds, epsilon, seed, max_error, ave_error])

    cols = ['data', 'generator', 'stats', 'T', 'epsilon' , 'seed', 'max error' , 'l1 error']
    Results_df = pd.DataFrame(Results, columns=cols)
    Results_df.to_csv('acsreal_results.csv', index=False)

if __name__ == "__main__":


    parser = argparse.ArgumentParser(prog='ACSreal Experiment',
                                     description='Run algorithm PrivGA and RAP++ on ACSreal data')

    parser.add_argument('--algo', choices=['PrivGA', 'RAP++'], default=['PrivGA', 'RAP++'], nargs='+')
    parser.add_argument('--queries', choices=['Halfspaces', 'Prefix', 'Ranges'], default=['Prefix', 'Halfspaces'], nargs='+')
    parser.add_argument('--epsilon', type=float, default=[0.07, 0.23, 0.52, 0.74, 1], nargs='+')
    parser.add_argument('--seed', type=int, default=[0, 1, 2], nargs='+')
    parser.add_argument('--rounds',  type=int, default=[25, 50, 75, 100], nargs='+')

    args = parser.parse_args()

    run_all_acs(args.algo, args.queries, args.epsilon, args.seed, args.rounds)