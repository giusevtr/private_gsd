import jax.numpy as jnp
import jax
import numpy as np

class PrivateMarginalsState:
    def __init__(self):
        self.NUM_STATS = 0
        self.true_stats = []
        self.priv_stats = []
        self.priv_marginals_fn = []
        self.priv_diff_marginals_fn = []
        self.selected_marginals = []

        self.priv_loss_l2_fn_jit_list = []
        self.priv_loss_l2_fn_jit_vmap_list = []

    def add_stats(self, true_stat, priv_stat, marginal_fn, diff_marginal_fn):
        self.NUM_STATS += true_stat.shape[0]
        self.true_stats.append(true_stat)
        self.priv_stats.append(priv_stat)
        self.priv_marginals_fn.append(marginal_fn)
        self.priv_diff_marginals_fn.append(diff_marginal_fn)

        # priv_loss_fn = lambda X: jnp.linalg.norm(priv_stat - marginal_fn(X), ord=2) / priv_stat.shape[0]
        priv_loss_fn = lambda X: jnp.linalg.norm(priv_stat - marginal_fn(X), ord=2)
        priv_loss_fn_jit = jax.jit(priv_loss_fn)
        self.priv_loss_l2_fn_jit_list.append(priv_loss_fn_jit)

        priv_loss_fn_jit_vmap = jax.vmap(priv_loss_fn_jit, in_axes=(0, ))
        self.priv_loss_l2_fn_jit_vmap_list.append(jax.jit(priv_loss_fn_jit_vmap))
        # fitness_fn_vmap = lambda x_pop: compute_error_vmap(x_pop)
        # fitness_fn_vmap_jit = jax.jit(fitness_fn)

    def get_true_stats(self):
        return jnp.concatenate(self.true_stats)

    def get_priv_stats(self):
        return jnp.concatenate(self.priv_stats)

    def get_stats(self, X):
        return jnp.concatenate([fn(X) for fn in self.priv_marginals_fn])

    def get_diff_stats(self, X):
        return jnp.concatenate([diff_fn(X) for diff_fn in self.priv_diff_marginals_fn])

    def true_loss_inf(self, X):
        true_stats_concat = jnp.concatenate(self.true_stats)
        sync_stats_concat = self.get_stats(X)
        return jnp.abs(true_stats_concat - sync_stats_concat).max()

    def true_loss_l2(self, X):
        true_stats_concat = jnp.concatenate(self.true_stats)
        # loss = jnp.sum(jnp.array([fn_jit(X)  for fn_jit in self.priv_loss_l2_fn_jit]))
        sync_stats_concat = self.get_stats(X)
        return jnp.linalg.norm(true_stats_concat - sync_stats_concat, ord=2) / true_stats_concat.shape[0]

    def priv_loss_inf(self, X):
        priv_stats_concat = jnp.concatenate(self.priv_stats)
        sync_stats_concat = self.get_stats(X)
        return jnp.abs(priv_stats_concat - sync_stats_concat).max()

    def priv_loss_l2(self, X):
        priv_stats_concat = jnp.concatenate(self.priv_stats)
        sync_stats_concat = self.get_stats(X)
        return jnp.linalg.norm(priv_stats_concat - sync_stats_concat, ord=2) / priv_stats_concat.shape[0]

    def priv_loss_l2_jit(self, X):
        loss = 0
        for jit_fn in self.priv_loss_l2_fn_jit_list:
            loss += jit_fn(X)
        return loss / self.NUM_STATS


    def priv_marginal_loss_l2_jit(self, X):
        losses = []
        # for jit_fn in self.priv_loss_l2_fn_jit_list:
        for i in range(len(self.priv_loss_l2_fn_jit_list)):
            stat_size = self.priv_stats[i].shape[0]
            jit_fn = self.priv_loss_l2_fn_jit_list[i]
            losses.append(jit_fn(X)/stat_size)
        return losses

    def priv_loss_l2_vmap_jit(self, X_pop):
        loss = None
        for jit_fn in self.priv_loss_l2_fn_jit_vmap_list:
            this_loss = jit_fn(X_pop)
            loss = this_loss if loss is None else loss + this_loss
        return loss / self.NUM_STATS if loss is not None else 0


    def priv_diff_loss_inf(self, X_oh):
        priv_stats_concat = jnp.concatenate(self.priv_stats)
        sync_stats_concat = self.get_diff_stats(X_oh)
        return jnp.abs(priv_stats_concat - sync_stats_concat).max()

    def priv_diff_loss_l2(self, X_oh):
        priv_stats_concat = jnp.concatenate(self.priv_stats)
        sync_stats_concat = self.get_diff_stats(X_oh)
        return jnp.linalg.norm(priv_stats_concat - sync_stats_concat, ord=2) ** 2 / priv_stats_concat.shape[0]



def get_private_statistics(self, key, rho):

    state = PrivateMarginalsState()

    rho_per_stat = rho / len(self.true_stats)
    for i in range(len(self.true_stats)):
        true_stat = self.true_stats[i]
        key, key_gaussian = jax.random.split(key, 2)
        sensitivity = self.sensitivity[i]

        sigma_gaussian = float(np.sqrt(sensitivity ** 2 / (2 * rho_per_stat)))
        gau_noise = jax.random.normal(key_gaussian, shape=true_stat.shape) * sigma_gaussian
        priv_stat = jnp.clip(true_stat + gau_noise, 0, 1)

        state.add_stats(true_stat, priv_stat, self.marginals_fn[i],
                        self.diff_marginals_fn[i] if not self.IS_REAL_VALUED else None)

    return state


def private_select_measure_statistic(self, key, rho_per_round, X_sync, state: PrivateMarginalsState):
    rho_per_round = rho_per_round / 2

    key, key_em = jax.random.split(key, 2)
    errors = self.get_sync_data_errors(X_sync)
    max_sensitivity = max(self.sensitivity)
    worse_index = exponential_mechanism(key_em, errors, jnp.sqrt(2 * rho_per_round), max_sensitivity)

    key, key_gaussian = jax.random.split(key, 2)
    selected_true_stat = self.true_stats[worse_index]

    sensitivity = self.sensitivity[worse_index]
    sigma_gaussian = float(np.sqrt(sensitivity ** 2 / (2 * rho_per_round)))
    gau_noise = jax.random.normal(key_gaussian, shape=selected_true_stat.shape) * sigma_gaussian

    selected_priv_stat = jnp.clip(selected_true_stat + gau_noise, 0, 1)

    state.add_stats(selected_true_stat, selected_priv_stat, self.marginals_fn[worse_index],
                    self.diff_marginals_fn[worse_index] if not self.IS_REAL_VALUED else None)

    return state

def exponential_mechanism(key:jnp.ndarray, scores: jnp.ndarray, eps0: float, sensitivity: float):
    dist = jax.nn.softmax(2 * eps0 * scores / (2 * sensitivity))
    cumulative_dist = jnp.cumsum(dist)
    max_query_idx = jnp.searchsorted(cumulative_dist, jax.random.uniform(key, shape=(1,)))
    return max_query_idx[0]
