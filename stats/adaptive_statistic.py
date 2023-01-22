import chex
import jax.numpy as jnp
import jax
import numpy as np


class AdaptiveStatisticState:
    private_statistics: list
    adaptive_rounds_count: int

    def __init__(self, statistic_module):
        self.private_statistics = []
        self.STAT_MODULE = statistic_module
        self.adaptive_rounds_count = 0

    def add_stats(self, stat_id, priv_stat):
        stat_id = int(stat_id)
        self.private_statistics.append((stat_id, priv_stat))

        # Need to sort by stat_id to maintain consistency
        sort_fn = lambda tup: tup[0]
        self.private_statistics.sort(key=sort_fn)

    def private_select_measure_statistic(self, key: chex.PRNGKey, rho_per_round: float, sync_data_mat: chex.Array,
                                         sample_num=1):
        self.adaptive_rounds_count += 1
        rho_per_round = rho_per_round / 2
        STAT = self.STAT_MODULE
        selected_stats = jnp.array(self.get_statistics_ids()).astype(int)

        errors = STAT.get_sync_data_errors(sync_data_mat)
        # max_sensitivity = max(self.STAT_MODULE.sensitivity)
        key, key_g = jax.random.split(key, 2)
        rs = np.random.RandomState(key_g)
        # Computing Gumbel noise scale based on: https://differentialprivacy.org/one-shot-top-k/

        noise = rs.gumbel(scale=(np.sqrt(sample_num) / (np.sqrt(2 * rho_per_round) * STAT.N)), size=errors.shape)
        # noise = rs.gumbel(scale=(np.sqrt(sample_num) / (np.sqrt(2 * rho_per_round) * STAT.N)), size=errors.shape)
        noise = jnp.array(noise)
        errors_noise = errors + noise
        errors_noise = errors_noise.at[selected_stats].set(-100000)
        top_k_indices = (-errors_noise).argsort()[:sample_num]
        for worse_index in top_k_indices:
            gaussian_rho_per_round = rho_per_round / sample_num
            key, key_gaussian = jax.random.split(key, 2)
            selected_true_stat = STAT.get_true_stat([worse_index])
            sensitivity = STAT.sensitivity[worse_index]
            sigma_gaussian = float(np.sqrt(sensitivity ** 2 / (2 * gaussian_rho_per_round)))
            gau_noise = jax.random.normal(key_gaussian, shape=selected_true_stat.shape) * sigma_gaussian
            selected_priv_stat = jnp.clip(selected_true_stat + gau_noise, 0, 1)
            self.add_stats(worse_index, selected_priv_stat)

    def private_measure_all_statistics(self, key: chex.PRNGKey, rho: float):
        self.adaptive_rounds_count += 1
        STAT = self.STAT_MODULE
        m = self.STAT_MODULE.get_num_queries()

        rho_per_marginal = rho / m
        for stat_id in range(m):
            key, key_gaussian = jax.random.split(key, 2)
            # selected_true_stat = STAT.true_stats[stat_id]
            selected_true_stat = STAT.get_true_stat([stat_id])
            sensitivity = STAT.sensitivity[stat_id]

            sigma_gaussian = float(np.sqrt(sensitivity ** 2 / (2 * rho_per_marginal)))
            gau_noise = jax.random.normal(key_gaussian, shape=selected_true_stat.shape) * sigma_gaussian
            selected_priv_stat = jnp.clip(selected_true_stat + gau_noise, 0, 1)
            self.add_stats(stat_id, selected_priv_stat)

    def get_true_statistics(self):
        # return jnp.concatenate([self.STAT_MODULE.get_true_stat(i) for i in self.statistics_ids])
        return self.STAT_MODULE.get_true_stat(self.get_statistics_ids())

    def get_statistics_ids(self):
        return [tup[0] for tup in self.private_statistics ]

    def get_private_statistics(self):
        return jnp.concatenate([priv_stat[1] for priv_stat in self.private_statistics])

def exponential_mechanism(key: jnp.ndarray, scores: jnp.ndarray, eps0: float, sensitivity: float):
    dist = jax.nn.softmax(2 * eps0 * scores / (2 * sensitivity))
    cumulative_dist = jnp.cumsum(dist)
    max_query_idx = jnp.searchsorted(cumulative_dist, jax.random.uniform(key, shape=(1,)))
    return max_query_idx[0]
