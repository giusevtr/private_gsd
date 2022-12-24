import jax
# from stats_v3 import PrivateStatistic
from stats_v3 import Marginals, PrivateMarginalsState
import time
from utils import Dataset, Domain
from utils.cdp2adp import cdp_rho
import numpy as np
import jax.numpy as jnp
import pandas as pd
from typing import Callable


class Generator:
    data_size: int

    def fit(self, key: jax.random.PRNGKeyArray, stat_module: PrivateMarginalsState, init_X=None, tolerance=0) -> Dataset:
        pass

    def fit_dp(self, key: jax.random.PRNGKeyArray, stat_module: PrivateMarginalsState, epsilon, delta, init_X=None, tolerance=0):

        rho = cdp_rho(epsilon, delta)
        return self.fit_zcdp(key, stat_module, rho, init_X, tolerance)

    def fit_zcdp(
            self,
            key: jax.random.PRNGKeyArray,
            stat_module: PrivateMarginalsState,
            rho: float,
            init_X=None, tolerance=0
    ):

        key, key_fit = jax.random.split(key, 2)
        dataset: Dataset
        sync_dataset = self.fit(key_fit, stat_module, init_X)
        return sync_dataset



    # @staticmethod
    # def default_debug_fn(X):
    def fit_dp_adaptive(self, key: jax.random.PRNGKeyArray, stat_module: Marginals, rounds, epsilon, delta, tolerance=0,
                        start_X=False,
                        print_progress=False,
                        debug_fn: Callable = None):
        rho = cdp_rho(epsilon, delta)
        return self.fit_zcdp_adaptive(key, stat_module, rounds, rho, tolerance, start_X, print_progress, debug_fn)

    def fit_zcdp_adaptive(self, key: jax.random.PRNGKeyArray, stat_module: Marginals, rounds, rho, tolerance=0,
                          start_X=False,
                             print_progress=False,
                          debug_fn: Callable = None):
        rho_per_round = rho / rounds
        domain = stat_module.domain

        key, key_init = jax.random.split(key, 2)
        X_sync = Dataset.synthetic_jax_rng(domain, N=self.data_size, rng=key_init)
        sync_dataset = None


        # true_answers = prefix_fn(data.to_numpy())

        ADA_DATA = {'epoch': [],
                    'average error': [],
                    'max error': [],
                    }

        true_stats = stat_module.get_true_stats()

        stat_state = PrivateMarginalsState()
        for i in range(1, rounds + 1):
            stime = time.time()

            # Select a query with max error using the exponential mechanism and evaluate
            key, subkey_select = jax.random.split(key, 2)
            stat_state = stat_module.private_select_measure_statistic(subkey_select, rho_per_round, X_sync, stat_state)
            # state = stat_module.priv_update(subkey_select, state, rho_per_round, X_sync)


            key, key_fit = jax.random.split(key, 2)
            dataset: Dataset
            if start_X:
                sync_dataset = self.fit(key_fit, stat_state, X_sync, tolerance=tolerance)
            else:
                sync_dataset = self.fit(key_fit, stat_state, tolerance=tolerance)

            ##### PROJECT STEP
            X_sync = sync_dataset.to_numpy()

            # Get errors for debugging
            errors_post_max = stat_module.get_sync_data_errors(X_sync).max()
            errors_post_avg = jnp.linalg.norm(true_stats - stat_module.get_stats(sync_dataset), ord=1)/true_stats.shape[0]

            if print_progress:
                gaussian_error = jnp.abs(stat_state.get_priv_stats() - stat_state.get_true_stats()).max()
                print(f'Epoch {i:03}: Total average error is {errors_post_avg:.6f}.\t Total max error is {errors_post_max:.5f}.'
                      f'\tRound max error = {stat_state.true_loss_inf(X_sync):.4f}.'
                      f'\tRound Gaussian max error {gaussian_error:.4f}.'
                      f'\tElapsed time = {time.time() - stime:.4f}s')
            if debug_fn is not None:
                debug_fn(i, X_sync)
            ADA_DATA['epoch'].append(i)
            ADA_DATA['average error'].append(errors_post_avg)
            ADA_DATA['max error'].append(errors_post_max)
            # ADA_DATA['round init error'].append(initial_max_error)

        df = pd.DataFrame(ADA_DATA)
        df['algo'] = str(self)
        self.ADA_DATA = df
        return sync_dataset

def exponential_mechanism(key:jnp.ndarray, scores: jnp.ndarray, eps0: float, sensitivity: float):
    dist = jax.nn.softmax(2 * eps0 * scores / (2 * sensitivity))
    cumulative_dist = jnp.cumsum(dist)
    max_query_idx = jnp.searchsorted(cumulative_dist, jax.random.uniform(key, shape=(1,)))
    return max_query_idx[0]

