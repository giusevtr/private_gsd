import itertools
import os
import jax
from models import PrivGAfast, SimpleGAforSyncDataFast, RelaxedProjectionPP
from stats import Halfspace, Prefix, Marginals
import time
from utils.utils_data import get_data
import pandas as pd
import argparse


def run_acs_example(generator,
                    data,
                    stats_module,
                    epsilon=1.00,
                    seed=0,
                    rounds=30,
                    num_sample=10,
                    adaptive=False):



    print(f'Running {generator} with epsilon={epsilon}, train module is {stats_module}')
    if adaptive:
        print(f'Adaptive with {rounds} rounds and {num_sample} samples.')
    else:
        print('Non-adaptive')
    stats_module.fit(data)
    folder = f'sync_data/{str(generator)}/{str(stats_module)}/{rounds}/{num_sample}/{epsilon:.2f}'
    os.makedirs(folder, exist_ok=True)
    path = f'{folder}/sync_data_{seed}.csv'

    print(f'Starting {generator}:')
    stime = time.time()
    key = jax.random.PRNGKey(seed)

    delta = 1 / len(data.df_real) ** 2
    if adaptive:
        sync_data = generator.fit_dp_adaptive(key, stat_module=stats_module, epsilon=epsilon, delta=delta,
                                              rounds=rounds, print_progress=True, num_sample=num_sample)
    else:
        sync_data = generator.fit_dp(key, stat_module=stats_module, epsilon=epsilon, delta=delta)

    print(f'Saving in {path}')
    sync_data.df_real.to_csv(path, index=False)

    errors = jax.numpy.abs(stats_module.get_true_stats() - stats_module.get_stats_jit(sync_data))
    ave_error = jax.numpy.linalg.norm(errors, ord=1)/errors.shape[0]
    print(f'{str(generator)}: Train max error = {errors.max():.4f}, ave_error={ave_error:.6f}, time={time.time() - stime:.4f}')
    return errors.max(), ave_error



def run_all_acs(gen_name, queries, epsilon_list: list, seed_list: list,
                adaptive: bool, rounds_list:list, samples_per_round_list,
                print_progress=False):

    data_name = f'folktables_2018_real_CA'
    data = get_data(f'{data_name}-mixed-train',
                    domain_name=f'domain/{data_name}-mixed', root_path='data_files/folktables_datasets_real')

    algos = {
        'PrivGA': PrivGAfast(num_generations=100000, print_progress=print_progress, strategy=SimpleGAforSyncDataFast(
            domain=data.domain, data_size=2000, population_size=100, elite_size=5, muta_rate=1, mate_rate=1)),
        'RAP++': RelaxedProjectionPP(domain=data.domain, data_size=1000, learning_rate=(0.0005,), print_progress=print_progress)
    }
    algo = algos[gen_name]

    modules = {
        'Halfspaces': Halfspace.get_kway_random_halfspaces(data.domain, k=1, rng=jax.random.PRNGKey(0), random_hs=150000)[0],
        'Prefix': Prefix.get_kway_prefixes(data.domain, k=1, rng=jax.random.PRNGKey(0), random_prefixes=200000)[0],
        'Ranges': Marginals.get_all_kway_mixed_combinations(data.domain, k_disc=1, k_real=2, bins=[2, 4, 8, 16, 32, 64])[0],
        '2-way Marginals': Marginals.get_all_kway_combinations(data.domain, k=2, bins=[2, 4, 8, 16, 32, 64])[0]
    }
    train_module = modules[queries]
    Results = []

    if adaptive:
        rounds_list = [1]
        samples_per_round_list = [0]
    for rounds, samples_per_round, epsilon, seed, in itertools.product(rounds_list, samples_per_round_list, epsilon_list, seed_list):

        max_error, ave_error = run_acs_example(algo, data,
                        stats_module=train_module,
                        epsilon=epsilon,
                        seed=seed,
                        rounds=rounds,
                        num_sample=samples_per_round,
                        adaptive=adaptive)
        Results.append([data_name, gen_name, queries, rounds, samples_per_round, epsilon, seed, max_error, ave_error])

    cols = ['data', 'generator', 'stats', 'T', 'samples', 'epsilon', 'seed', 'max error', 'l1 error']
    Results_df = pd.DataFrame(Results, columns=cols)
    print(f'Saving ', f'acsreal_{gen_name}_{queries}_results.csv')
    Results_df.to_csv(f'acsreal_{gen_name}_{queries}_results.csv', index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='ACSreal Experiment',
                                     description='Run algorithm PrivGA and RAP++ on ACSreal data')

    parser.add_argument('--algo', choices=['PrivGA', 'RAP++'], default='PrivGA')
    parser.add_argument('--queries', choices=['Halfspaces', 'Prefix', 'Ranges'], default='Prefix')
    parser.add_argument('--epsilon', type=float, default=[1], nargs='+')
    parser.add_argument('--seed', type=int, default=[0], nargs='+')
    parser.add_argument('-a', '--adaptive', action='store_true', default=True)  # on/off flag
    parser.add_argument('--rounds',  type=int, default=[50], nargs='+')
    parser.add_argument('--samples_per_round',  type=int, default=[10], nargs='+')

    args = parser.parse_args()

    run_all_acs(args.algo, args.queries, args.epsilon, args.seed, args.adaptive, args.rounds, args.samples_per_round)