import jax
import time
import jax.numpy as jnp
from typing import Callable
from genetic_sd.adaptive_statistics import AdaptiveChainedStatistics
from genetic_sd.utils import Dataset
from snsynth.utils import cdp_rho


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

    def fit(self, key, stat: AdaptiveChainedStatistics, init_data: Dataset = None,
            tolerance: float = 0, adaptive_epoch: int = 1) -> Dataset:
        pass

    def fit_dp(self, key, stat_module: AdaptiveChainedStatistics, epsilon: float, delta: float,
               init_data: Dataset = None, tolerance: float = 0) -> Dataset:
        rho = cdp_rho(epsilon, delta)
        return self.fit_zcdp(key, stat_module, rho, init_data, tolerance)

    def fit_zcdp(self, key, stat_module: AdaptiveChainedStatistics, rho: float,
                 init_data: Dataset = None, tolerance: float = 0) -> Dataset:
        key_stats, key_fit = jax.random.split(key)
        stat_module.reselect_stats()
        stat_module.private_measure_all_statistics(key_stats, rho)

        return self.fit(key_fit, stat_module, init_data, tolerance, adaptive_epoch=1)


    def fit_dp_adaptive(self, key,
                        stat_module: AdaptiveChainedStatistics, rounds,
                        epsilon: float, delta: float,
                        tolerance: float = 0,
                        start_sync=True,
                        print_progress=True,
                        debug_fn: Callable = None, num_sample=1):
        rho = cdp_rho(epsilon, delta)
        return self.fit_zcdp_adaptive(key, stat_module, rounds, rho, tolerance, start_sync, print_progress, debug_fn,
                                      num_sample)

    def fit_zcdp_adaptive(self, key,
                          stat_module: AdaptiveChainedStatistics,
                          rounds: int,
                          rho: float, tolerance: float = 0,
                          start_sync=False,
                          print_progress=False,
                          debug_fn: Callable = None, num_sample=1):

        # Reset selected statistics
        stat_module.reselect_stats()

        rho_per_round = rho / rounds

        key, key_init = jax.random.split(key, 2)
        init_seed = int(jax.random.randint(key_init, minval=0, maxval=2 ** 20, shape=(1,))[0])
        sync_dataset = Dataset.synthetic(stat_module.get_domain(), N=self.data_size, seed=init_seed, null_values=0.02)

        for i in range(1, rounds + 1):
            if i < rounds:
                self.loss_change_threshold = 0.01
            else:
                self.loss_change_threshold = 0.001

            # Select a query with max error using the exponential mechanism and evaluate
            select_time = time.time()
            # X_sync = sync_dataset.to_numpy()
            key, subkey_select = jax.random.split(key, 2)
            stat_module.private_select_measure_statistic(subkey_select, rho_per_round, sync_dataset, num_sample)
            select_time = time.time() - select_time

            fit_time = time.time()
            key, key_fit = jax.random.split(key, 2)
            dataset: Dataset
            if start_sync:
                new_sync_dataset = self.fit(key_fit, stat_module, sync_dataset, tolerance=tolerance, adaptive_epoch=i)
            else:
                new_sync_dataset = self.fit(key_fit, stat_module, tolerance=tolerance, adaptive_epoch=i)
            fit_time = time.time() - fit_time

            if print_progress:
                # Errors of selected statistics. Debug the success of the project step.
                priv_stats = stat_module.get_selected_noised_statistics()
                selected_true_stats = stat_module.get_selected_statistics_without_noise()
                stat_fn = stat_module.get_selected_dataset_statistics_fn()
                init_round_errors = jnp.abs(selected_true_stats - stat_fn(sync_dataset))
                round_errors = jnp.abs(selected_true_stats - stat_fn(new_sync_dataset))
                gau_error = jnp.abs(selected_true_stats - priv_stats)

                # Get errors for debugging. This is
                all_true_stats = stat_module.get_all_true_statistics()
                all_sync_stat_fn = stat_module.get_dataset_statistics_fn()
                all_sync_stats = all_sync_stat_fn(new_sync_dataset)
                all_errors = jnp.abs(all_true_stats - all_sync_stats)

                print(f'Epoch {i:03}: '
                      f'\tTotal: error(max/avg) is {all_errors.max():.4f}/{all_errors.mean():.7f}.\t ||'
                      f'\tRound init: True error(max/l2) is {init_round_errors.max():.5f}/{init_round_errors.mean():.7f}.'
                      f'\tRound final: True error(max/l2) is {round_errors.max():.5f}/{round_errors.mean():.7f}.'
                      f'\tGaussian error(max/l2) is {gau_error.max():.5f}/{gau_error.mean():.7f}.'
                      f'\tElapsed time(fit/select)={fit_time:>7.3f}/{select_time:.3f}')


            sync_dataset = new_sync_dataset

            if debug_fn is not None:
                debug_fn(i, sync_dataset)

        return sync_dataset
