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
    def fit(self, key, true_stats, stat_module, init_X=None) -> Dataset:
        pass

    def fit_privately(
            self,
            # data: Dataset,
            stat_module: Statistic,
            epsilon: float,
            seed: int = 0,
    ):
        key = jax.random.PRNGKey(seed)
        # X = data.to_numpy()
        stime = time.time()
        n, dim = X.shape

        n = X.shape[0]
        delta = 1 / (n ** 2)
        rho = cdp_rho(epsilon, delta)
        # rng = np.random.default_rng(seed)

        # Get statistics and noisy statistics
        stat_module = stat_module
        stats_fn = stat_module.get_stats_fn()
        true_stats = stats_fn(X)

        num_queries = true_stats.shape[0]
        sensitivity = stat_module.get_sensitivity() / n
        # print(f'Running {str(generator)}:')
        # print(f'Stats is {str(stat_module)}')
        # print(f'dimension = {dim}, num queries = {num_queries}, sensitivity={sensitivity:.4f}')

        # sigma_gaussian = float(np.sqrt(num_queries / (2 * (n ** 2) * rho)))
        sigma_gaussian = float(np.sqrt(sensitivity ** 2 / (2 * rho)))

        key, key_gaussian = jax.random.split(key, 2)
        jax.random.normal(key_gaussian, shape=true_stats.shape)
        true_stats_noise = true_stats + jax.random.normal(key_gaussian, shape=true_stats.shape) * sigma_gaussian
        gaussian_error = jnp.abs(true_stats - true_stats_noise)
        # print(f'epsilon={epsilon}, '
        #       f'Gaussian error: L1 {jnp.linalg.norm(gaussian_error, ord=1):.5f},'
        #       f'Gaussian error: L2 {jnp.linalg.norm(gaussian_error, ord=2):.5f},'
        #       f' Max = {gaussian_error.max():.5f}')

        key, key_fit = jax.random.split(key, 2)
        dataset: Dataset
        sync_dataset = self.fit(key_fit, true_stats_noise, stat_module)

        # X_sync = sync_dataset.to_numpy()
        # elapsed_time = time.time() - stime
        # Use large statistic set to evaluate the final synthetic data.
        # evaluate_stats_fn = stat_module.get_stats_fn(num_cols=dim, num_rand_queries=10000, seed=0)
        # evaluate_true_stats = stats_fn(X)
        # sync_stats = stats_fn(X_sync)
        # errors = jnp.abs(evaluate_true_stats - sync_stats)
        # error = jnp.linalg.norm(errors, ord=1)
        # error_l2 = jnp.linalg.norm(errors, ord=2)
        # max_error = errors.max()
        # print(f'Final L1 error = {error:.5f}, L2 error = {error_l2:.5f},  max error ={max_error:.5f}\n')
        # return X_sync, error, max_error, elapsed_time
        return sync_dataset

    def fit_privately_adaptive(self, stat_module: Statistic, rounds, seed, epsilon, delta=1e-5):
        """
        PRIVACY NOT YET IMPLEMENTED
        """
        key = jax.random.PRNGKey(seed)

        rho = cdp_rho(epsilon, delta)
        rho_per_round = rho / rounds
        # sigma = np.sqrt(0.5 / (alpha*rho_per_round))
        # exp_eps = np.sqrt(8*(1-alpha)*rho_per_round)

        domain = stat_module.domain

        rng = np.random.default_rng(0)
        key, key_init = jax.random.split(key, 2)
        X_sync = Dataset.synthetic_jax_rng(domain, N=self.data_size, rng=key_init)
        data_sync = None

        stat_fn = stat_module.get_stats_fn()
        selected_indices = []
        selected_indices_jnp = jnp.array(selected_indices)

        # true_answers = prefix_fn(data.to_numpy())
        true_answers = stat_module.get_true_stats()

        DATA = {'epoch': [], 'average error': [], 'max error': []}
        key = jax.random.PRNGKey(seed)
        for i in range(1, rounds + 1):
            key, key_sub = jax.random.split(key, 2)

            errors = stat_module.get_sync_data_errors(X_sync)
            initial_max_error = errors.max()

            if len(selected_indices) > 0:
                errors.at[selected_indices_jnp].set(-100000)

            worse_index = errors.argmax()

            selected_indices.append(worse_index)
            selected_indices_jnp = jnp.array(selected_indices)

            # fit synthetic data to selected statistics
            sub_stat_module = stat_module.get_sub_stat_module(selected_indices)
            sub_true_answers_jnp = sub_stat_module.get_true_stats()
            data_sync = self.fit(key_sub, sub_true_answers_jnp, sub_stat_module)
            X_sync = data_sync.to_numpy()

            # Get errors for debugging
            errors_post_epoch = stat_module.get_sync_data_errors(X_sync)
            total_max_error = errors_post_epoch.max()
            round_max_error = errors_post_epoch[jnp.array(selected_indices)].max()

            # Get average error for debugging
            average_error = jnp.sum(jnp.abs(true_answers - stat_fn(X_sync))) / true_answers.shape[0]
            print(f'epoch {i:03}. Total average error is {average_error:.6f}.\t Total max error is {total_max_error:.5f}.'
                f'\tRound init max-error is {initial_max_error:.4f} and final max-error is {round_max_error:.4f}')

            DATA['epoch'].append(i)
            DATA['average error'].append(average_error)
            DATA['max error'].append(total_max_error)

            # est = engine.estimate(measurements, total)

        df = pd.DataFrame(DATA)
        df['algo'] = str(self)
        self.DATA = df
        return data_sync




