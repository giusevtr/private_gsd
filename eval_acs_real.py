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



def eval_all_acs(generators_list, queries_list, epsilon_list: list, seed_list: list, rounds_list: list,
                 samples_per_round_list: list):

    Results = []
    data_name = f'folktables_2018_real_CA'
    data = get_data(f'{data_name}-mixed-train',
                    domain_name=f'domain/{data_name}-mixed', root_path='data_files/folktables_datasets_real')
    modules = {
            'Halfspaces': Halfspace.get_kway_random_halfspaces(data.domain, k=1, rng=jax.random.PRNGKey(0), random_hs=20000)[0],
            'Prefix': Prefix.get_kway_prefixes(data.domain, k=1, rng=jax.random.PRNGKey(0), random_prefixes=20000)[0],
            'Ranges': Marginals.get_all_kway_mixed_combinations(data.domain, k_disc=1, k_real=2, bins=[2, 4, 8, 16, 32, 64])[0]
        }
    for query_set in queries_list:

        train_module = modules[query_set]
        print(f'fitting data on module {query_set}...', end=' ')
        train_module.fit(data)
        print(f'Done fitting.')

        for gen_name, rounds, samples_per_round, epsilon, seed in itertools.product(generators_list,
                                                                                rounds_list,
                                                                                    samples_per_round_list,
                                                                                    epsilon_list, seed_list):

            sync_path = f'sync_data/{gen_name}/{query_set}/{rounds}/{samples_per_round}/{epsilon:.2f}/sync_data_{seed}.csv'
            if not os.path.exists(sync_path):
                print(f'\tCannot find {sync_path}')
                continue
            print(f'reading {sync_path}')
            df = pd.read_csv(sync_path, index_col=None)
            sync_data = Dataset(df, data.domain)

            errors = jax.numpy.abs(train_module.get_true_stats() - train_module.get_stats_jit(sync_data))
            max_error = errors.max()
            ave_error = errors.mean()
            print(f'{gen_name}, {query_set}, {rounds}, {samples_per_round}., {epsilon}: '
                  f'Train max error = {max_error:.4f}, ave_error={ave_error:.6f}')
            Results.append(
                [data_name, gen_name, query_set, rounds, samples_per_round, epsilon, seed, max_error, ave_error])

            Results.append([data_name, gen_name, query_set, rounds, epsilon, seed, max_error, ave_error])

    cols = ['data', 'generator', 'stats', 'T', 'samples', 'epsilon', 'seed', 'max error', 'l1 error']
    # cols = ['data', 'generator', 'stats', 'T', 'epsilon' , 'seed', 'max error' , 'l1 error']
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
    parser.add_argument('--samples_per_round',  type=int, default=[10], nargs='+')


    args = parser.parse_args()

    eval_all_acs(generators_list=args.algo,
                 queries_list=args.queries,
                 epsilon_list=args.epsilon,
                 seed_list=args.seed,
                 rounds_list=args.rounds,
                 samples_per_round_list=args.samples_per_round)