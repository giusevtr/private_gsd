import jax.random
import numpy as np
import os

import pandas as pd

# from mbi import Dataset, GraphicalModel, FactoredInference
from utils import Dataset
from scipy.special import softmax
# from cdp2adp import cdp_rho
from utils.cdp2adp import  cdp_rho
from stats_v2 import TwoWayPrefix
import jax.numpy as jnp
from models_v2 import PrivGA
from utils.plot_low_dim_data import plot_2d_data


def worst_approximated(workload_answers, data_sync: Dataset, workload, eps, penalty=True):
    """ Select a (noisy) worst-approximated marginal for measurement.

    :param workload_answers: a dictionary of true answers to the workload
        keys are cliques
        values are numpy arrays, corresponding to the counts in the marginal
    :param est: a GraphicalModel object that approximates the data distribution
    :param: workload: The list of candidates to consider in the exponential mechanism
    :param eps: the privacy budget to use for this step.
    """
    errors = np.array([])
    for cl in workload:
        sz = data_sync.domain.size(cl)
        bias = data_sync.domain.size(cl) if penalty else 0
        x = workload_answers[cl]
        xest = data_sync.project(cl).datavector() / data_sync.df.shape[0]
        this_error = np.abs(x - xest).sum( )
        max_error = np.abs(x - xest).max( )
        errors = np.append(errors, this_error - bias)
        # print(f'marginal = {cl}, avg error = {this_error/sz:.4f}, max error = {max_error}')
    sensitivity = 2.0
    prob = softmax(0.5 * eps / sensitivity * (errors - errors.max()))
    key = np.random.choice(len(errors), p=prob)
    return workload[key]


def adaptive(generator, prefix_queries: TwoWayPrefix, data, epsilon, rounds, seed, data_size=500, delta=1e-5):
    """
    Implementation of MWEM + PGM

    :param data: an mbi.Dataset object
    :param epsilon: privacy budget
    :param delta: privacy parameter (ignored)
    :param workload: A list of cliques (attribute tuples) to include in the workload (default: all pairs of attributes)
    :param rounds: The number of rounds of MWEM to run (default: number of attributes)
    :param maxsize_mb: [New] a limit on the size of the model (in megabytes), used to filter out candidate cliques from selection.
        Used to avoid MWEM+PGM failure modes (intractable model sizes).
        Set to np.inf if you would like to run MWEM as originally described without this modification
        (Note it may exceed resource limits if run for too many rounds)

    Implementation Notes:
    - During each round of MWEM, one clique will be selected for measurement, but only if measuring the clique does        not increase size of the graphical model too much
    """

    N = len(data.df)
    rho = cdp_rho(epsilon, delta)
    rho_per_round = rho / rounds
    # sigma = np.sqrt(0.5 / (alpha*rho_per_round))
    # exp_eps = np.sqrt(8*(1-alpha)*rho_per_round)

    domain = data.domain

    rng = np.random.default_rng(0)
    data_sync = Dataset.synthetic_rng(domain, N=data_size, rng=rng)

    prefix_fn = prefix_queries.get_stats_fn()
    selected_indices = []
    selected_indices_jnp = jnp.array(selected_indices)

    true_answers = prefix_fn(data.to_numpy())
    sub_true_answers = []

    DATA = {'epoch': [], 'l1': [], 'max': []}
    key = jax.random.PRNGKey(seed)
    for i in range(1, rounds+1):
        key, key_sub = jax.random.split(key, 2)

        round_answers = prefix_fn(data_sync.to_numpy())
        errors = jnp.abs(true_answers - round_answers)

        if len(selected_indices) > 0:
            errors.at[selected_indices_jnp].set(-100000)

        worse_index = errors.argmax()
        print(f'sected query:', prefix_queries.columns[worse_index, :], prefix_queries.thresholds[worse_index,: ])
        sub_true_answers.append(true_answers[worse_index])

        selected_indices.append(worse_index)
        selected_indices_jnp = jnp.array(selected_indices)

        sub_prefix_module = prefix_queries.get_sub_stat_module(selected_indices_jnp)

        # generator = get_generator(data.domain, marginal_queries, data_size=data_size, seed=seed)
        sub_true_answers_jnp = jnp.array(sub_true_answers)
        data_sync = generator.fit(key_sub, sub_true_answers_jnp, sub_prefix_module, init_sync=data_sync.to_numpy())

        os.makedirs('progress', exist_ok=True)
        save_dir = f'progress/{str(generator)}'
        os.makedirs(save_dir, exist_ok=True)
        plot_2d_data(data_array=data_sync.to_numpy(), title=f'epoch = {i}', save_path=f'{save_dir}/{i:004}.png')

        sync_ans = prefix_fn(data_sync.to_numpy())
        m = true_answers.shape[0]
        errors = jnp.abs(true_answers - sync_ans)
        m_r = selected_indices_jnp.shape[0]
        round_errors = errors[selected_indices_jnp]
        print(f'epoch {i:03}. Total round l1/max error = {np.linalg.norm(errors, ord=1)/m:.3f}/{np.max(np.abs(errors)):.3f}.'
              f'\t\tRound l1/max error = {jnp.linalg.norm(round_errors, ord=11)/m_r:.3f}/{round_errors.max():.3f}')

        DATA['epoch'].append(i)
        DATA['l1'].append(np.linalg.norm(errors, ord=1)/m)
        DATA['max'].append(errors.max())

        # est = engine.estimate(measurements, total)

    df = pd.DataFrame(DATA)
    df['algo'] = str(generator)
    return data_sync,  df


from toy_datasets.sparse import get_sparse_dataset
if __name__ == "__main__":

    # data = get_data('adult', 'adult-mini', root_path='../data_files/')
    data = get_sparse_dataset(DATA_SIZE=1000)

    # plot_2d_data(data.to_numpy(), title='original')
    POPSIZE = 100
    DATA_SIZE = 200
    num_generations=300
    get_gen_list = [
        # ('RP++', RelaxedProjectionPP(domain=data.domain,  data_size=DATA_SIZE,
        #         iterations=1000, learning_rate=(0.0005, 0.001, 0.005, 0.01, 0.05))),
        ('PrivGA', PrivGA(domain=data.domain, data_size=200, num_generations=num_generations,
            popsize=POPSIZE, top_k=30, stop_loss_time_window=50, print_progress=False, start_mutations=32,
            )),
        ('PrivGA(reg)', PrivGA(domain=data.domain, data_size=200, num_generations=num_generations,
            popsize=POPSIZE, top_k=30, stop_loss_time_window=50, print_progress=False, start_mutations=32,
            reg_prefix_queries=TwoWayPrefix.get_stat_module(domain=data.domain, num_rand_queries=100, seed=0)
            ))
    ]

    # prefix_queries = TwoWayPrefix(domain=data.domain, num_rand_queries=1000, seed=0)
    prefix_queries = TwoWayPrefix.get_stat_module(domain=data.domain, num_rand_queries=1000, seed=0)
    stat_fn = prefix_queries.get_stats_fn()
    true_stats = stat_fn(data.to_numpy())

    results = []
    for name, gen in get_gen_list:
        for seed in [0]:
            print(f'Running {name}', str(gen))
            data_sync, info_df = adaptive(gen, prefix_queries, data, epsilon=1, rounds=40, seed=seed, data_size=200)
            info_df.to_csv(f'results/{str(gen)}.csv', index=False)
            results.append(info_df)
            sync_stats = stat_fn(data_sync.to_numpy())
            errors = jnp.abs(true_stats - sync_stats)
            print(f'l1 error = {jnp.linalg.norm(errors, ord=1)/errors.shape[0]:.4f}, max={jnp.max(errors):.4f}')
            print()

    df_all = pd.concat(results)
    df_all.to_csv('results/results.csv', index=False)
