import numpy as np
import itertools
# from mbi import Dataset, GraphicalModel, FactoredInference
from utils import Dataset, Domain
from scipy.special import softmax
from scipy import sparse
# from cdp2adp import cdp_rho
from utils.cdp2adp import  cdp_rho
import argparse
from utils.utils_data import get_data
from stats import Marginals
from models import Generator
import jax.numpy as jnp
from models import PrivGA, RelaxedProjection


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


def adaptive(get_generator, data, epsilon, seed, delta=1e-5, workload=None, rounds=None, alpha=0.5, data_size=500):
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
    - During each round of MWEM, one clique will be selected for measurement, but only if measuring the clique does
        not increase size of the graphical model too much
    """
    if workload is None:
        workload = list(itertools.combinations(data.domain, 2))
    if rounds is None:
        rounds = len(data.domain)

    N = len(data.df)
    rho = cdp_rho(epsilon, delta)
    rho_per_round = rho / rounds
    sigma = np.sqrt(0.5 / (alpha*rho_per_round))
    exp_eps = np.sqrt(8*(1-alpha)*rho_per_round)
    marginal_sensitivity = np.sqrt(2)/N

    domain = data.domain
    workload_answers = {cl: data.project(cl).datavector()/N for cl in workload}
    measurements = []
    cliques = []

    rng = np.random.default_rng(0)
    data_sync = Dataset.synthetic_rng(domain, N=data_size, rng=rng)

    for i in range(1, rounds+1):
        # [New] Only consider candidates that keep the model sufficiently small
        candidates = [cl for cl in workload if cl not in cliques]
        ax = worst_approximated(workload_answers, data_sync, candidates, exp_eps)
        n = domain.size(ax)
        x = data.project(ax).datavector() / N
        y = x + np.random.normal(loc=0, scale=marginal_sensitivity*sigma, size=n)
        measurements.append(y)


        init_sync_ans = data_sync.project(ax).datavector() / data_sync.df.shape[0]
        start_error = np.linalg.norm(x - init_sync_ans, ord=1)
        print('Round', i, 'of', rounds, 'Selected', ax, f': start round l1-error {start_error:.3f}', end=' ')

        priv_true_stats = jnp.concatenate(measurements)

        cliques.append(ax)
        marginal_queries = Marginals(domain, cliques)
        generator = get_generator(data.domain, marginal_queries, data_size=data_size, seed=seed)

        data_sync = generator.fit(priv_true_stats, init_X=data_sync.to_numpy())

        sync_ans = data_sync.project(ax).datavector() / data_sync.df.shape[0]
        print(f'final round l1-error = {np.linalg.norm(x - sync_ans, ord=1):.3f}, max-error ={np.max(np.abs(x - sync_ans)):.3f}')

        # est = engine.estimate(measurements, total)

    return data_sync


if __name__ == "__main__":

    data = get_data('adult', 'adult-mini', root_path='../data_files/')

    get_gen_list = [
        ('RP(0.005)', RelaxedProjection.get_generator(learning_rate=0.005)),
        ('RP(0.05)', RelaxedProjection.get_generator(learning_rate=0.05)),
        ('PrivGA', PrivGA.get_generator(popsize=1000,
                             top_k=5,
                             num_generations=300,
                             stop_loss_time_window=50,
                             print_progress=False,
                             start_mutations=32))
    ]

    marginals = Marginals.get_all_kway_combinations(domain=data.domain, k=2)
    marginal_stat_fn = marginals.get_stats_fn()
    true_stats = marginal_stat_fn(data.to_numpy())
    for name, gen in get_gen_list:
        print(f'Running {name}')
        data_sync = adaptive(gen, data, epsilon=1, seed=0, data_size=200)
        sync_stats = marginal_stat_fn(data_sync.to_numpy())
        errors = jnp.abs(true_stats - sync_stats)
        print(f'l1 error = {jnp.linalg.norm(errors, ord=1):.4f}, max={jnp.max(errors):.4f}')
        print()
