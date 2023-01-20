import chex
import jax.numpy as jnp
import jax
import numpy as np

from stats import Marginals

class AdaptiveStatisticState:
    statistics_ids: list
    private_statistics: list
    private_stat_fn: list
    private_stat_fn_jit: list
    private_diff_stat_fn_jit: list

    def __init__(self, statistic_module: Marginals):
        self.statistics_ids = []
        self.private_statistics = []
        self.private_stat_fn = []
        self.private_stat_fn_jit = []
        self.private_diff_stat_fn_jit = []
        self.statistic_fn_jit_dict = {}
        self.INFO = {}
        self.STAT_MODULE = statistic_module
        self.NUM_STATS = 0
        self.priv_loss_l2_fn_jit_list = []
        self.priv_population_l2_loss_fn_list = []

    def add_stats(self, stat_id, priv_stat):
        stat_id = int(stat_id)

        # true_stat = self.STAT_MODULE.get_true_stat(stat_id)
        # marginal_fn = self.STAT_MODULE.marginals_fn_jit[stat_id]
        # marginal_fn = jax.jit(self.STAT_MODULE.get_stat_fn(stat_id))
        # self.NUM_STATS += true_stat.shape[0]
        self.statistics_ids.append(stat_id)
        self.private_statistics.append(priv_stat)
        # self.private_stat_fn.append(self.STAT_MODULE.get_stat_fn(stat_id))
        # self.private_stat_fn_jit.append(jax.jit(self.STAT_MODULE.get_stat_fn(stat_id)))
        # self.private_diff_stat_fn_jit.append(jax.jit(self.STAT_MODULE.get_diff_stat_fn(stat_id)))
        self.statistic_fn_jit_dict[stat_id] = {}

        # priv_loss_fn = lambda X: jnp.linalg.norm(priv_stat - marginal_fn(X), ord=2) / priv_stat.shape[0]
        # priv_loss_fn = lambda X: jnp.linalg.norm(priv_stat - marginal_fn(X), ord=2)
        # priv_loss_fn_jit = jax.jit(priv_loss_fn)
        # self.priv_loss_l2_fn_jit_list.append(priv_loss_fn_jit)
        # priv_population_loss_fn_jit_vmap = jax.vmap(priv_loss_fn_jit, in_axes=(0, ))
        # self.priv_population_l2_loss_fn_list.append(jax.jit(priv_population_loss_fn_jit_vmap))

    def private_select_measure_statistic(self, key: chex.PRNGKey, rho_per_round: float, sync_data_mat: chex.Array):
        rho_per_round = rho_per_round / 2
        STAT = self.STAT_MODULE

        key, key_em = jax.random.split(key, 2)
        errors = STAT.get_sync_data_errors(sync_data_mat)
        max_sensitivity = max(self.STAT_MODULE.sensitivity)
        worse_index = exponential_mechanism(key_em, errors, jnp.sqrt(2 * rho_per_round), 1 / STAT.N)

        key, key_gaussian = jax.random.split(key, 2)
        # selected_true_stat = STAT.true_stats[worse_index]
        selected_true_stat = STAT.get_true_stat([worse_index])

        sensitivity = STAT.sensitivity[worse_index]
        sigma_gaussian = float(np.sqrt(sensitivity ** 2 / (2 * rho_per_round)))
        gau_noise = jax.random.normal(key_gaussian, shape=selected_true_stat.shape) * sigma_gaussian
        selected_priv_stat = jnp.clip(selected_true_stat + gau_noise, 0, 1)
        self.add_stats(worse_index, selected_priv_stat)

    def private_measure_all_statistics(self, key: chex.PRNGKey, rho: float):
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
        return self.STAT_MODULE.get_true_stat(self.statistics_ids)

    def get_private_statistics(self):
        return jnp.concatenate(self.private_statistics)

    def private_statistics_fn(self, X):
        # return self.STAT_MODULE.get_stats(X, self.statistics_ids)
        return jnp.concatenate([stat_fn(X) for stat_fn in self.private_stat_fn])
        # return jnp.concatenate([stat_fn(X) for stat_fn in self.private_stat_fn])

    def private_statistics_fn_jit(self, X):
        return jnp.concatenate([stat_fn_jit(X) for stat_fn_jit in self.private_stat_fn_jit])

    # def private_diff_statistics_fn(self, X, sigmoid=None):
    #     return jnp.concatenate([self.STAT_MODULE.diff_marginals_fn[i](X, sigmoid) for i in self.statistics_ids])

    # def private_diff_statistics_fn_jit(self, X, sigmoid=None):
    #     return jnp.concatenate([diff_stat_fn_jit(X, sigmoid) for diff_stat_fn_jit in self.private_diff_stat_fn_jit])
        # return jnp.concatenate([self.STAT_MODULE.diff_marginals_fn_jit[i](X, sigmoid) for i in self.statistics_ids])

    def true_loss_inf(self, X):
        true_stats_concat = self.get_true_statistics()
        sync_stats_concat = self.private_statistics_fn_jit(X)
        return jnp.abs(true_stats_concat - sync_stats_concat).max()

    def true_loss_l2(self, X):
        true_stats_concat =  self.get_true_statistics()
        sync_stats_concat = self.private_statistics_fn_jit(X)
        return jnp.linalg.norm(true_stats_concat - sync_stats_concat, ord=2) / true_stats_concat.shape[0]

    def private_loss_inf(self, X):
        priv_stats_concat = self.get_private_statistics()
        sync_stats_concat = self.private_statistics_fn_jit(X)
        return jnp.abs(priv_stats_concat - sync_stats_concat).max()

    def private_loss_l2(self, X):
        priv_stats_concat = self.get_private_statistics()
        sync_stats_concat = self.private_statistics_fn_jit(X)
        return jnp.linalg.norm(priv_stats_concat - sync_stats_concat, ord=2) / priv_stats_concat.shape[0]

    def private_population_l2_loss_fn_jit(self, X_pop):
        loss = None
        for jit_fn in self.priv_population_l2_loss_fn_list:
            this_loss = jit_fn(X_pop)
            loss = this_loss if loss is None else loss + this_loss
        # return loss / self.NUM_STATS if loss is not None else 0
        return loss if loss is not None else 0


    # def private_diff_loss_inf(self, X_oh):
    #     priv_stats_concat = self.get_private_statistics()
    #     sync_stats_concat = self.private_diff_statistics_fn_jit(X_oh)
    #     return jnp.abs(priv_stats_concat - sync_stats_concat).max()

    # def private_diff_loss_l2(self, X_oh):
    #     priv_stats_concat = self.get_private_statistics()
    #     sync_stats_concat = self.private_diff_statistics_fn_jit(X_oh)
    #     return jnp.linalg.norm(priv_stats_concat - sync_stats_concat, ord=2) ** 2 / priv_stats_concat.shape[0]


def exponential_mechanism(key:jnp.ndarray, scores: jnp.ndarray, eps0: float, sensitivity: float):
    dist = jax.nn.softmax(2 * eps0 * scores / (2 * sensitivity))
    cumulative_dist = jnp.cumsum(dist)
    max_query_idx = jnp.searchsorted(cumulative_dist, jax.random.uniform(key, shape=(1,)))
    return max_query_idx[0]

