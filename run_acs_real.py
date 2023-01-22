import itertools
import os
import jax
from models import PrivGAfast, SimpleGAforSyncDataFast, RelaxedProjectionPP
from stats import Halfspace, Prefix, Marginals
import time
from utils.utils_data import get_data
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
    data_name = f'folktables_2018_real_CA'
    folder = f'sync_data/{str(algo)}/{stats_module}/{data_name}/{epsilon:.2f}/{rounds}'
    os.makedirs(folder, exist_ok=True)
    path = f'{folder}/sync_data_{seed}.csv'

    print(f'Starting {algo}:')
    stime = time.time()
    key = jax.random.PRNGKey(seed)

    if adaptive:
        sync_data = algo.fit_dp_adaptive(key, stat_module=stats_module,  epsilon=epsilon, delta=1e-6,
                                        rounds=rounds, print_progress=True, num_sample=num_sample)
    else:
        sync_data = algo.fit_dp(key, stat_module=stats_module,  epsilon=epsilon, delta=1e-6)

    print(f'Saving in {path}')
    sync_data.df.to_csv(path, index=False)

    errors = jax.numpy.abs(stats_module.get_true_stats() - stats_module.get_stats_jit(sync_data))
    ave_error = jax.numpy.linalg.norm(errors, ord=1)/errors.shape[0]
    print(f'{str(algo)}: Train max error = {errors.max():.4f}, ave_error={ave_error:.6f}, time={time.time()-stime:.4f}')



def run_all_acs(algo, queries, epsilon: list, seed: list, adaptive: bool, rounds:list, samples_per_round):

    data_name = f'folktables_2018_real_CA'
    data = get_data(f'{data_name}-mixed-train',
                    domain_name=f'domain/{data_name}-mixed', root_path='data_files/folktables_datasets_real')

    algos = {
        'privga': PrivGAfast(num_generations=100000, print_progress=False, strategy=SimpleGAforSyncDataFast(
            domain=data.domain, data_size=2000, population_size=100, elite_size=5, muta_rate=1, mate_rate=1)),
        'rap++': RelaxedProjectionPP(domain=data.domain, data_size=1000, learning_rate=(0.01,), print_progress=False)
    }
    algo = algos[algo]

    modules = {
        'halfspaces': Halfspace.get_kway_random_halfspaces(data.domain, k=1, rng=jax.random.PRNGKey(0), random_hs=20000)[0],
        'prefix': Prefix.get_kway_prefixes(data.domain, k=1, rng=jax.random.PRNGKey(0), random_prefixes=20000)[0],
        'ranges': Marginals.get_all_kway_mixed_combinations(data.domain, k_disc=1, k_real=2, bins=[2, 4, 8, 16, 32, 64])[0]
    }
    train_module = modules[queries]

    for epsilon, seed, rounds, samples_per_round in itertools.product(epsilon, seed, rounds, samples_per_round):

        run_acs_example(algo, data,
                        stats_module=train_module,
                        epsilon=epsilon,
                        seed=seed,
                        rounds=rounds,
                        num_sample=samples_per_round,
                        adaptive=adaptive)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='ACSreal Experiment',
                                     description='Run algorithm PrivGA and RAP++ on ACSreal data')

    parser.add_argument('--algo', choices=['privga', 'rap++'], default='privga')
    parser.add_argument('--queries', choices=['halfspaces', 'prefix', 'ranges'], default='prefix')
    parser.add_argument('--epsilon', type=float, default=[1], nargs='+')
    parser.add_argument('--seed', type=int, default=[0], nargs='+')
    parser.add_argument('-a', '--adaptive', action='store_true', default=True)  # on/off flag
    parser.add_argument('--rounds',  type=int, default=[50], nargs='+')
    parser.add_argument('--samples_per_round',  type=int, default=[10], nargs='+')

    args = parser.parse_args()

    run_all_acs(args.algo, args.queries, args.epsilon, args.seed, args.adaptive, args.rounds, args.samples_per_round)