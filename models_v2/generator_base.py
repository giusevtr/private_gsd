import jax
from stats_v2 import Statistic
import time
from utils import Dataset, Domain
from utils.cdp2adp import cdp_rho
import numpy as np
import jax.numpy as jnp
import pandas as pd
from typing import Callable

class Generator:

    data_size: int
    def fit(self, key: jax.random.PRNGKeyArray, true_stats: jnp.ndarray, stat_module: Statistic, init_X=None, confidence_bound=None) -> Dataset:
        pass

    def fit_dp(self, key: jax.random.PRNGKeyArray, stat_module: Statistic, epsilon, delta, init_X=None):

        rho = cdp_rho(epsilon, delta)
        return self.fit_zcdp(key, stat_module, rho, init_X)

    def fit_zcdp(
            self,
            key: jax.random.PRNGKeyArray,
            stat_module: Statistic,
            rho: float,
            init_X=None,
            confidence_bound=None
    ):
        # Get statistics and noisy statistics
        true_stats = stat_module.get_true_stats()
        sensitivity = stat_module.get_sensitivity()
        sigma_gaussian = float(np.sqrt(sensitivity ** 2 / (2 * rho)))

        key, key_gaussian = jax.random.split(key, 2)
        true_stats_noise = true_stats + jax.random.normal(key_gaussian, shape=true_stats.shape) * sigma_gaussian
        key, key_fit = jax.random.split(key, 2)
        dataset: Dataset
        sync_dataset = self.fit(key_fit, true_stats_noise, stat_module, init_X,confidence_bound)
        return sync_dataset

    def fit_dp_adaptive(self, key: jax.random.PRNGKeyArray, stat_module: Statistic, rounds, epsilon, delta, print_progress=False,
                        debug_fn: Callable = None):
        rho = cdp_rho(epsilon, delta)
        return self.fit_zcdp_adaptive(key, stat_module, rounds, rho, print_progress, debug_fn)

    def fit_zcdp_adaptive(self, key: jax.random.PRNGKeyArray, stat_module: Statistic, rounds, rho, print_progress=False,
                          debug_fn: Callable = None):
        """
        PRIVACY NOT YET IMPLEMENTED
        """
        rho_per_round = rho / rounds
        domain = stat_module.domain
        sensitivity = stat_module.get_sensitivity()
        sigma_gaussian = float(np.sqrt(sensitivity ** 2 / (2 * rho_per_round)))
        gau_confidence_bound = jnp.sqrt(2 * sigma_gaussian ** 2 * jnp.log(rounds / 0.01))
        confidence_bound = (gau_confidence_bound + 0.5 / self.data_size)/3
        confidence_bound = None
        # print(f'confidence_bound:{confidence_bound}')
        key, key_init = jax.random.split(key, 2)
        X_sync = Dataset.synthetic_jax_rng(domain, N=self.data_size, rng=key_init)
        data_sync = None

        stat_fn = stat_module.get_stats_fn()
        selected_indices = []
        selected_indices_jnp = jnp.array(selected_indices)

        # true_answers = prefix_fn(data.to_numpy())
        true_answers = stat_module.get_true_stats()

        ADA_DATA = {'epoch': [],
                    'average error': [],
                    'max error': [],
                    'round init error': [],
                    'round max error': []}
        for i in range(1, rounds + 1):
            key, key_sub = jax.random.split(key, 2)

            errors = stat_module.get_sync_data_errors(X_sync)
            initial_max_error = errors.max()

            if len(selected_indices) > 0:
                errors.at[selected_indices_jnp].set(-100000)

            ####################################
            ## REPLACE WITH EXPONENTIAL MECHANISM
            ####################################
            key, key_em = jax.random.split(key, 2)
            worse_index = exponential_mechanism(key_em, errors, jnp.sqrt(rho_per_round),stat_module.get_sensitivity())
            # worse_index = errors.argmax()
            # print(f'Debug: {i}) selected marginal is ', stat_module.kway_combinations[worse_index], end=' ')

            selected_indices.append(worse_index)
            selected_indices_jnp = jnp.array(selected_indices)

            # fit synthetic data to selected statistics
            sub_stat_module = stat_module.get_sub_stat_module(selected_indices)
            data_sync = self.fit_zcdp(key_sub, sub_stat_module, rho_per_round/2, init_X=X_sync, confidence_bound=confidence_bound)
            X_sync = data_sync.to_numpy()

            # Get errors for debugging
            errors_post_epoch = stat_module.get_sync_data_errors(X_sync)
            total_max_error = errors_post_epoch.max()
            round_max_error = errors_post_epoch[selected_indices_jnp].max()

            # Get average error for debugging
            average_error = jnp.sum(jnp.abs(true_answers - stat_fn(X_sync))) / true_answers.shape[0]
            if print_progress:
                print(f'epoch {i:03}. Total average error is {average_error:.6f}.\t Total max error is {total_max_error:.5f}.'
                    f'\tRound init max-error is {initial_max_error:.4f} and final max-error is {round_max_error:.4f}')
            if debug_fn is not None:
                debug_fn(X_sync)
            ADA_DATA['epoch'].append(i)
            ADA_DATA['average error'].append(average_error)
            ADA_DATA['max error'].append(total_max_error)
            ADA_DATA['round init error'].append(initial_max_error)
            ADA_DATA['round max error'].append(round_max_error)

        df = pd.DataFrame(ADA_DATA)
        df['algo'] = str(self)
        self.ADA_DATA = df
        return data_sync

    # @staticmethod
    # def default_debug_fn(X):


def exponential_mechanism(key:jnp.ndarray, scores: jnp.ndarray, eps0: float, sensitivity: float):
    dist = jax.nn.softmax(2 * eps0 * scores / (2 * sensitivity))
    cumulative_dist = jnp.cumsum(dist)
    max_query_idx = jnp.searchsorted(cumulative_dist, jax.random.uniform(key, shape=(1,)))
    return max_query_idx[0]

