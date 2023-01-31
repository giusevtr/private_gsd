import itertools
import os
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from models import PrivGAfast, SimpleGAforSyncDataFast, RelaxedProjectionPP
from stats import Halfspace, Prefix, Marginals
from toy_datasets.circles import get_circles_dataset
from toy_datasets.moons import get_moons_dataset
from toy_datasets.sparse import get_sparse_dataset
from toy_datasets.digits import get_digits_dataset
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from utils import Dataset
import time
from utils.plot_low_dim_data import plot_2d_data


def run_toy_example(
                    data: Dataset,
                    algo_name,
                    queries_name,
                    epsilon=1.00,
                    seed=0,
                    rounds=30,
                    num_sample=5,
                    adaptive=False,
                    debug_fn=None):

    print(f'Running {algo_name} with epsilon={epsilon}, train module is {queries_name}')
    if adaptive:
        print(f'Adaptive with {rounds} rounds and {num_sample} samples.')
    else:
        print('Non-adaptive')


    ALGO = {
        'PrivGA': PrivGAfast(num_generations=10000, print_progress=True, strategy=SimpleGAforSyncDataFast(
            domain=data.domain, data_size=2000, population_size=100, elite_size=5, muta_rate=1, mate_rate=1)),
        'RAP++': RelaxedProjectionPP(domain=data.domain, data_size=1000, learning_rate=(0.01,), print_progress=False)
    }
    algo = ALGO[algo_name]

    modules = {
        'Halfspaces': Halfspace.get_kway_random_halfspaces(data.domain, k=1, rng=jax.random.PRNGKey(0), random_hs=2000)[0],
        'Prefix': Prefix.get_kway_prefixes(data.domain, k=1, rng=jax.random.PRNGKey(0), random_prefixes=2000)[0],
        'Ranges': Marginals.get_all_kway_combinations(data.domain, k=3, bins=[2, 4, 8, 16, 32])[0]
    }
    train_module = modules[queries_name]
    train_module.fit(data)



    print(f'Starting {algo}:')
    stime = time.time()
    key = jax.random.PRNGKey(seed)

    delta = 1 / len(data.df)**2
    if adaptive:
        sync_data = algo.fit_dp_adaptive(key, stat_module=train_module,  epsilon=epsilon, delta=delta,
                                        rounds=rounds, print_progress=True, num_sample=num_sample, debug_fn=debug_fn)
        debug_fn(1, sync_data)
    else:
        sync_data = algo.fit_dp(key, stat_module=train_module,  epsilon=epsilon, delta=delta)
        debug_fn(1, sync_data)

    # debug_fn(1, sync_data)
    errors = jax.numpy.abs(train_module.get_true_stats() - train_module.get_stats_jit(sync_data))
    ave_error = jax.numpy.linalg.norm(errors, ord=1)/errors.shape[0]
    print(f'{str(algo)}: Train max error = {errors.max():.4f}, ave_error={ave_error:.6f}, time={time.time()-stime:.4f}')
    return sync_data





DATASETS = {
    'circles': get_circles_dataset(DATA_SIZE=50000),
    'sparse': get_sparse_dataset(DATA_SIZE=50000),
    'moons': get_moons_dataset(DATA_SIZE=50000),
    'digits': get_digits_dataset(),
    # 'ACSreal' :
}

def debug_fn(t, tempdata):
    X = tempdata.to_numpy()
    if t == 0:
        save_path = f'{folder}/{data_name}_original.png'
    else:
        save_path = f'{folder}/img_{t:04}.png'

    plot_2d_data(tempdata.to_numpy(), title=f'epoch={t}:' , alpha=0.9, save_path=save_path)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ACSreal Experiment',
                                     description='Run algorithm PrivGA and RAP++ on ACSreal data')

    parser.add_argument('--data', choices=['circles', 'sparse', 'moons', 'digits'], default='circles')
    parser.add_argument('--algo', choices=['PrivGA', 'RAP++'], default='PrivGA')
    parser.add_argument('--queries', choices=['Halfspaces', 'Prefix', 'Ranges'], default='Prefix')
    parser.add_argument('--epsilon', type=float, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-a', '--adaptive', action='store_true')  # on/off flag
    parser.add_argument('--rounds',  type=int, default=50)
    parser.add_argument('--samples_per_round',  type=int, default=1)

    args = parser.parse_args()

    data_name = args.data
    algo_name = args.algo
    queries_name = args.queries
    epsilon = args.epsilon
    seed = args.seed
    adaptive = args.adaptive
    rounds = args.rounds
    samples_per_round = args.samples_per_round


    data = DATASETS[data_name]

    if adaptive:
        folder = f'sync_data/{data_name}/{algo_name}/{queries_name}/{rounds:03}/{samples_per_round:03}/{epsilon:.2f}/{seed}'
    else:
        folder = f'sync_data/{data_name}/{algo_name}/{queries_name}/nonadaptive/{epsilon:.2f}/{seed}'

    # print(f'Saving in {folder}'
    os.makedirs(folder, exist_ok=True)
    sync_data_path = f'{folder}/sync_data.csv'

    def debug_fn(t, tempdata: Dataset):
        if data_name == 'digits':
            os.makedirs(f'{folder}/images_epoch={t}', exist_ok=True)
            D = tempdata.to_numpy()
            x = np.array(D[:, :-1])
            y = D[:, -1].squeeze()
            for i in range(20):
                plt.title(f'label={int(y[i])}')
                plt.gray()
                plt.matshow(x[i, :].reshape((8, 8)))
                plt.savefig(f'{folder}/images/img_{i}.png')
                plt.close()
            tempdata.df.to_csv(f'{folder}/sync_data.csv', index=False)
        else:
            if t == 0:
                save_path = f'{folder}/{data_name}_original.png'
            else:
                save_path = f'{folder}/img_{t:04}.png'
            # tempdata.df.to_csv()
            # sns.scatterplot(data=tempdata.df, x='A', y='B', hue='C')
            print(f'saving {save_path}')
            plot_2d_data(tempdata.to_numpy(), title=f'epoch={t}:\nTrain error = ', alpha=0.9, save_path=save_path)

    sync_data: Dataset
    sync_data = run_toy_example(data,
                                algo_name=algo_name,
                    queries_name=queries_name,
                    epsilon=epsilon,
                    rounds=rounds,
                    num_sample=samples_per_round,
                    adaptive=adaptive,
                                debug_fn=debug_fn)

    folder = f'sync_data/{data_name}/{algo_name}/{queries_name}/{rounds}/{samples_per_round}/{epsilon}'
    os.makedirs(folder, exist_ok=True)
    sync_data.df.to_csv(f'{folder}/sync_data_{seed}.csv')
