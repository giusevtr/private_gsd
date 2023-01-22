import chex
import jax
# from stats import PrivateStatistic
from stats import Marginals, AdaptiveStatisticState
import time
from utils import Dataset, Domain, timer
from utils.cdp2adp import cdp_rho
import numpy as np
import jax.numpy as jnp
import pandas as pd
from typing import Callable
import matplotlib.pyplot as plt


class Generator:
    data_size: int
    early_stop_elapsed_time = 1
    last_time: float = None
    last_error: float = None
    loss_change_threshold: float = 0.001

    def early_stop_init(self):
        self.last_time: float = time.time()
        self.last_error = 10000000
        self.start_time = time.time()
        self.last_update_iteration = 0

    def early_stop(self, t, error):
        """
        :param t: current iteration
        :param error: the error on this iteration
        :return:
        """
        if self.last_update_iteration == 0:  # If it's the first time don't halt
            self.last_error = error
            self.last_update_iteration = t
            return False

        stop_early = False
        loss_change = (self.last_error - error) / self.last_error
        iterations_increase = (t - self.last_update_iteration) / self.last_update_iteration
        if loss_change > self.loss_change_threshold:
            # If the loss improves same the iteration and the error
            self.last_error = error
            self.last_update_iteration = t

        if iterations_increase > 0.1:
            # If the iterations have increased by more than 10% since the last improvement then halt
            stop_early = True
            self.last_error = error
            self.last_update_iteration = t
        return stop_early

    # def early_stop(self, t, error):
    #     current_time = time.time()
    #     stop_early = False
    #     if current_time - self.last_time > self.early_stop_elapsed_time:
    #         loss_change = jnp.abs(error - self.last_error) / self.last_error
    #         if loss_change < 0.001:
    #             stop_early = True
    #         self.last_time = current_time
    #         self.last_error = error
    #     return stop_early

    def fit(self, key: jax.random.PRNGKeyArray, stat: AdaptiveStatisticState, init_data: Dataset = None,
            tolerance: float = 0) -> Dataset:
        pass

    def fit_dp(self, key: jax.random.PRNGKeyArray, stat_module: Marginals, epsilon: float, delta: float,
               init_data: Dataset = None, tolerance: float = 0) -> Dataset:
        rho = cdp_rho(epsilon, delta)
        return self.fit_zcdp(key, stat_module, rho, init_data, tolerance)

    def fit_zcdp(self, key: jax.random.PRNGKeyArray, stat_module: Marginals, rho: float,
                 init_data: Dataset = None, tolerance: float = 0) -> Dataset:
        key_stats, key_fit = jax.random.split(key)
        stat = AdaptiveStatisticState(stat_module)
        stat.private_measure_all_statistics(key_stats, rho)
        return self.fit(key_fit, stat, init_data, tolerance)

    # @staticmethod
    # def default_debug_fn(X):
    def fit_dp_adaptive(self, key: jax.random.PRNGKeyArray, stat_module: Marginals, rounds,
                        epsilon: float, delta: float,
                        tolerance: float = 0,
                        start_sync=True,
                        print_progress=False,
                        debug_fn: Callable = None, num_sample=1):
        rho = cdp_rho(epsilon, delta)
        return self.fit_zcdp_adaptive(key, stat_module, rounds, rho, tolerance, start_sync, print_progress, debug_fn,
                                      num_sample)

    def fit_zcdp_adaptive(self, key: jax.random.PRNGKeyArray, stat_module: Marginals, rounds: int,
                          rho: float, tolerance: float = 0,
                          start_sync=False,
                          print_progress=False,
                          debug_fn: Callable = None, num_sample=1):
        rho_per_round = rho / rounds
        domain = stat_module.domain

        key, key_init = jax.random.split(key, 2)
        # X_sync = Dataset.synthetic_jax_rng(domain, N=self.data_size, rng=key_init)
        init_seed = int(jax.random.randint(key_init, minval=0, maxval=2 ** 20, shape=(1,))[0])
        sync_dataset = Dataset.synthetic(domain, N=self.data_size, seed=init_seed)
        # sync_dataset = None

        # true_answers = prefix_fn(data.to_numpy())

        ADA_DATA = {'epoch': [],
                    'average error': [],
                    'max error': [],
                    'round true max error': [],
                    'round true avg error': [],
                    'round priv max error': [],
                    'round priv avg error': [],
                    'time': [],
                    }

        true_stats = stat_module.get_true_stats()

        adaptive_statistic = AdaptiveStatisticState(stat_module)
        for i in range(1, rounds + 1):
            if i < rounds:
                self.loss_change_threshold = 0.01
            else:
                self.loss_change_threshold = 0.001

            stime = timer()

            # Select a query with max error using the exponential mechanism and evaluate
            select_time = timer()
            X_sync = sync_dataset.to_numpy()
            key, subkey_select = jax.random.split(key, 2)
            adaptive_statistic.private_select_measure_statistic(subkey_select, rho_per_round, X_sync, num_sample)
            select_time = timer() - select_time
            # state = stat_module.priv_update(subkey_select, state, rho_per_round, X_sync)

            fit_time = timer()
            key, key_fit = jax.random.split(key, 2)
            dataset: Dataset
            if start_sync:
                sync_dataset = self.fit(key_fit, adaptive_statistic, sync_dataset, tolerance=tolerance)
            else:
                sync_dataset = self.fit(key_fit, adaptive_statistic, tolerance=tolerance)
            fit_time = timer() - fit_time

            if print_progress:
                ##### PROJECT STEP
                X_sync = sync_dataset.to_numpy()

                # Round results
                priv_stats = adaptive_statistic.get_private_statistics()
                selected_true_stats = stat_module.get_true_stat(adaptive_statistic.get_statistics_ids())
                stat_fn = stat_module.get_stat_fn(adaptive_statistic.get_statistics_ids())
                sync_stats = stat_fn(X_sync)
                round_errors = jnp.abs(selected_true_stats-sync_stats)
                gau_error = jnp.abs(selected_true_stats - priv_stats)

                # Get errors for debugging
                all_true_stats = stat_module.get_true_stats()
                all_sync_stats = stat_module.get_stats_jax_jit(X_sync)
                all_errors = jnp.abs(all_true_stats - all_sync_stats)

                print(f'Epoch {i:03}: Total error(max/avg) is {all_errors.max():.4f}/{all_errors.mean():.7f}.\t ||'
                      f'\tRound: True error(max/l2) is {round_errors.max():.5f}/{round_errors.mean():.7f}.'
                      # f'\t(true) max error = {jnp.abs(true_stats - sync_stats).max():.4f}.'
                      # f'\t(true)  l2 error = {stat_state.true_loss_l2(X_sync):.5f}.'
                      f'\tGaussian error(max/l2) is {gau_error.max():.5f}/{gau_error.mean():.7f}.'
                      f'\tElapsed time(fit/select)={fit_time:>7.3f}/{select_time:.3f}')
            if debug_fn is not None:
                debug_fn(i, sync_dataset)
            # ADA_DATA['round init error'].append(initial_max_error)

        return sync_dataset


def exponential_mechanism(key: jnp.ndarray, scores: jnp.ndarray, eps0: float, sensitivity: float):
    dist = jax.nn.softmax(2 * eps0 * scores / (2 * sensitivity))
    cumulative_dist = jnp.cumsum(dist)
    max_query_idx = jnp.searchsorted(cumulative_dist, jax.random.uniform(key, shape=(1,)))
    return max_query_idx[0]
