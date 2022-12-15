import jax
from stats_v2 import Statistic
import time
from utils import Dataset, Domain
from utils.cdp2adp import cdp_rho
import numpy as np
import jax.numpy as jnp
import pandas as pd


class Generator:

    data_size: int
    def fit(self, key: jax.random.PRNGKeyArray, true_stats: jnp.ndarray, stat_module: Statistic, init_X=None) -> Dataset:
        pass

    def fit_dp(self, key: jax.random.PRNGKeyArray, stat_module: Statistic, epsilon, delta, init_X=None):

        rho = cdp_rho(epsilon, delta)
        return self.fit_zcdp(key, stat_module, rho, init_X)

    def fit_zcdp(
            self,
            key: jax.random.PRNGKeyArray,
            stat_module: Statistic,
            rho: float,
            init_X=None
    ):
        # Get statistics and noisy statistics
        true_stats = stat_module.get_true_stats()
        sensitivity = stat_module.get_sensitivity()
        sigma_gaussian = float(np.sqrt(sensitivity ** 2 / (2 * rho)))

        key, key_gaussian = jax.random.split(key, 2)
        true_stats_noise = true_stats + jax.random.normal(key_gaussian, shape=true_stats.shape) * sigma_gaussian
        key, key_fit = jax.random.split(key, 2)
        dataset: Dataset
        sync_dataset = self.fit(key_fit, true_stats_noise, stat_module, init_X)
        return sync_dataset

    def fit_dp_adaptive(self, key: jax.random.PRNGKeyArray, stat_module: Statistic, rounds, epsilon, delta, init_X=None):
        rho = cdp_rho(epsilon, delta)
        return self.fit_zcdp_adaptive(key, stat_module, rounds, rho, init_X)

    def fit_zcdp_adaptive(self, key: jax.random.PRNGKeyArray, stat_module: Statistic, rounds, rho, init_X=None):
        """
        PRIVACY NOT YET IMPLEMENTED
        """

        # rho = cdp_rho(epsilon, delta)
        rho_per_round = rho / rounds
        # sigma = np.sqrt(0.5 / (alpha*rho_per_round))
        # exp_eps = np.sqrt(8*(1-alpha)*rho_per_round)

        domain = stat_module.domain

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
            worse_index = errors.argmax()
            print(f'\t\tdebug: {i}) selected marginal is ', stat_module.kway_combinations[worse_index])

            selected_indices.append(worse_index)
            selected_indices_jnp = jnp.array(selected_indices)

            # fit synthetic data to selected statistics
            sub_stat_module = stat_module.get_sub_stat_module(selected_indices)
            data_sync = self.fit_zcdp(key_sub, sub_stat_module, rho_per_round, init_X=X_sync)
            X_sync = data_sync.to_numpy()

            # Get errors for debugging
            errors_post_epoch = stat_module.get_sync_data_errors(X_sync)
            total_max_error = errors_post_epoch.max()
            round_max_error = errors_post_epoch[jnp.array(selected_indices)].max()

            # Get average error for debugging
            average_error = jnp.sum(jnp.abs(true_answers - stat_fn(X_sync))) / true_answers.shape[0]
            print(f'epoch {i:03}. Total average error is {average_error:.6f}.\t Total max error is {total_max_error:.5f}.'
                f'\tRound init max-error is {initial_max_error:.4f} and final max-error is {round_max_error:.4f}')
            ADA_DATA['epoch'].append(i)
            ADA_DATA['average error'].append(average_error)
            ADA_DATA['max error'].append(total_max_error)
            ADA_DATA['round init error'].append(initial_max_error)
            ADA_DATA['round max error'].append(round_max_error)
            # ADA_DATA = {'epoch': [],
            #             'average error': [],
            #             'max error': [],
            #             'round init error': [],
            #             'round max error': []}
            # est = engine.estimate(measurements, total)

        df = pd.DataFrame(ADA_DATA)
        df['algo'] = str(self)
        self.ADA_DATA = df
        return data_sync




