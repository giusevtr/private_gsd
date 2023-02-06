import chex
import jax.numpy as jnp
import jax
import numpy as np

from typing import Callable
from utils import Dataset, Domain
from tqdm import tqdm


class AdaptiveStatisticState:
    all_workloads: list
    selected_workloads: list
    adaptive_rounds_count: int
    domain: Domain

    def fit(self, data: Dataset):
        X = data.to_numpy()
        self.domain = data.domain
        num_attrs = len(self.domain.attrs)
        self.N = X.shape[0]
        self.all_workloads = []
        self.selected_workloads = []
        # self.real_stats = []
        num_workloads = self.get_num_workloads()
        for i in tqdm(range(num_workloads), f'Fitting workloads. Data has {self.N} rows and {num_attrs} attributes.'):
            workload_fn = self._get_workload_fn(workload_ids=[i])
            stats = workload_fn(X)
            self.all_workloads.append((i, workload_fn, stats))

        self.all_statistics_fn = self._get_workload_fn()

    def get_statistics_ids(self):
        return [tup[0] for tup in self.selected_workloads]

    def get_all_true_statistics(self):
        return jnp.concatenate([tup[2] for tup in self.all_workloads])

    def get_all_statistics_fn(self):
        return self._get_workload_fn()

    def get_selected_noised_statistics(self):
        return jnp.concatenate([selected[2] for selected in self.selected_workloads])

    def get_selected_statistics_without_noise(self):
        return jnp.concatenate([selected[3] for selected in self.selected_workloads])

    def get_selected_statistics_fn(self):
        return self._get_workload_fn(self.get_statistics_ids())

    def get_domain(self):
        return self.domain

    def get_num_workloads(self) -> int:
        pass

    def _get_workload_fn(self, workload_ids: list = None) -> Callable:
        pass

    def _get_workload_sensitivity(self, workload_id: int = None, N: int = None) -> float:
        pass

    def _get_workload_positions(self, workload_id: int = None) -> tuple:
        pass

    # def add_stats(self, workload_id, workload_fn, noised_workload_statistics, true_workload_statistics):
    #     workload_id = int(workload_id)
    #     self.selected_workloads.append((workload_id, workload_fn, noised_workload_statistics, true_workload_statistics))

    # def private_measure_all_statistics(self, key: chex.PRNGKey, rho: float):
    #     # self.adaptive_rounds_count += 1
    #     self.selected_workloads = []
    #     m = self.get_num_workloads()
    #     rho_per_marginal = rho / m
    #     for stat_id in range(m):
    #         key, key_gaussian = jax.random.split(key, 2)
    #         _, stat_fn, stats = self.all_workloads[stat_id]
    #         sensitivity = self._get_workload_sensitivity(stat_id, self.N)
    #
    #         sigma_gaussian = float(np.sqrt(sensitivity ** 2 / (2 * rho_per_marginal)))
    #         gau_noise = jax.random.normal(key_gaussian, shape=stats.shape) * sigma_gaussian
    #         selected_noised_stat = jnp.clip(stats + gau_noise, 0, 1)
    #         self.add_stats(stat_id, stat_fn, selected_noised_stat, stats)

    # def private_select_measure_statistic(self, key: chex.PRNGKey, rho_per_round: float, sync_data_mat: chex.Array,
    #                                      sample_num=1):
    #     rho_per_round = rho_per_round / 2
    #     # STAT = self.STAT_MODULE
    #     selected_stats = jnp.array(self.get_statistics_ids()).astype(int)
    #
    #     errors = self.get_sync_data_errors(sync_data_mat)
    #     # max_sensitivity = max(self.STAT_MODULE.sensitivity)
    #     key, key_g = jax.random.split(key, 2)
    #     rs = np.random.RandomState(key_g)
    #     # Computing Gumbel noise scale based on: https://differentialprivacy.org/one-shot-top-k/
    #
    #     noise = rs.gumbel(scale=(np.sqrt(sample_num) / (np.sqrt(2 * rho_per_round) * self.N)), size=errors.shape)
    #     # noise = rs.gumbel(scale=(np.sqrt(sample_num) / (np.sqrt(2 * rho_per_round) * STAT.N)), size=errors.shape)
    #     noise = jnp.array(noise)
    #     errors_noise = errors + noise
    #     errors_noise = errors_noise.at[selected_stats].set(-100000)
    #     top_k_indices = (-errors_noise).argsort()[:sample_num]
    #     for worse_index in top_k_indices:
    #         gaussian_rho_per_round = rho_per_round / sample_num
    #         key, key_gaussian = jax.random.split(key, 2)
    #
    #         _, selected_workload_fn, selected_true_stat = self.all_workloads[worse_index]
    #         # selected_true_stat = STAT.get_true_stat([worse_index])
    #         sensitivity = self._get_workload_sensitivity(worse_index, self.N)
    #         sigma_gaussian = float(np.sqrt(sensitivity ** 2 / (2 * gaussian_rho_per_round)))
    #         gau_noise = jax.random.normal(key_gaussian, shape=selected_true_stat.shape) * sigma_gaussian
    #         selected_priv_stat = jnp.clip(selected_true_stat + gau_noise, 0, 1)
    #         self.add_stats(worse_index, selected_workload_fn, selected_priv_stat, selected_true_stat)


    # def get_sync_data_errors(self, X):
    #     max_errors = []
    #     for stat_id, workload_fn, statistics in self.all_workloads:
    #         workload_error = jnp.abs(statistics - workload_fn(X))
    #         max_errors.append(workload_error.max())
    #     return jnp.array(max_errors)



def exponential_mechanism(key: jnp.ndarray, scores: jnp.ndarray, eps0: float, sensitivity: float):
    dist = jax.nn.softmax(2 * eps0 * scores / (2 * sensitivity))
    cumulative_dist = jnp.cumsum(dist)
    max_query_idx = jnp.searchsorted(cumulative_dist, jax.random.uniform(key, shape=(1,)))
    return max_query_idx[0]
