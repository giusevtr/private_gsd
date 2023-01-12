import itertools
import jax
import jax.numpy as jnp
from utils import Dataset, Domain
from utils.utils_data import Domain
from stats.private_statistics import PrivateMarginalsState
from stats import Marginals
import numpy as np
import chex


class Halfspace(Marginals):
    true_stats: list
    marginals_fn: list
    row_answer_fn: list
    diff_marginals_fn: list
    sensitivity: list

    def __init__(self, domain: Domain, kway_combinations: list, num_random_halfspaces: int = 10):
        """
        :param domain:
        :param kway_combinations:
        :param num_random_halfspaces: number of random halfspaces for each marginal that contains a real-valued feature
        """
        super().__init__(domain, kway_combinations)
        self.num_hs = num_random_halfspaces

    def __str__(self):
        return f'Halfspaces'

    @staticmethod
    def get_all_kway_combinations(domain, k: int, random_hs: int = 10):
        kway_combinations = [list(idx) for idx in itertools.combinations(domain.attrs, k)]
        return Halfspace(domain, kway_combinations, num_random_halfspaces=random_hs)

    @staticmethod
    def get_all_kway_mixed_combinations_v1(domain, k, random_hs: int = 32):
        kway_combinations = []
        for cols in itertools.combinations(domain.attrs, k):
            count_real = 0
            for c in list(cols):
                count_real += 1 * (c in domain.get_numeric_cols())
            if 0 < count_real < k:
                kway_combinations.append(list(cols))
        return Halfspace(domain, kway_combinations, num_random_halfspaces=random_hs), kway_combinations


    def fit(self, data: Dataset, rng: chex.PRNGKey):
        self.true_stats = []
        self.marginals_fn = []
        self.diff_marginals_fn = []
        self.sensitivity = []

        X = data.to_numpy()
        self.N = X.shape[0]
        self.domain = data.domain

        for cols in self.kway_combinations:
            if self.is_workload_numeric(cols):
                rng, rng_sub = jax.random.split(rng)
                fn, sensitivity = self.get_halfspace_marginal_stats_fn_helper(cols, rng_sub, num_hs_projections=self.num_hs)
            else:
                fn, sensitivity = self.get_categorical_marginal_stats_fn_helper(cols)

            self.true_stats.append(fn(X))
            self.marginals_fn.append(jax.jit(fn))
            self.sensitivity.append(sensitivity / self.N)

            diff_fn, sensitivity = self.get_differentiable_stats_fn_helper(cols)
            self.diff_marginals_fn.append(diff_fn)

    # @jax.jit
    # def get_halfspace
    def get_halfspace_marginal_stats_fn_helper(self, cols: tuple, jax_rng: chex.PRNGKey, num_hs_projections):
        # Get categorical columns meta data
        sizes = []
        for col in self.domain.get_categorical_cols():
            sizes.append(self.domain.size(col))

        cat_cols, numeric_cols = [], []
        for col in cols:
            if col in self.domain.get_categorical_cols():
                cat_cols.append(col)
            else:
                numeric_cols.append(col)

        idx = self.domain.get_attribute_indices(cat_cols)
        dim = len(self.domain.attrs)
        sizes_jnp = [jnp.arange(s + 1).astype(float) for i, s in zip(idx, sizes)]

        for i in range(num_hs_projections):
            sizes_jnp.append(jnp.array([-1, 0.0, 1]))


        num_attrs_idx = self.domain.get_attribute_indices(numeric_cols)
        numeric_dim = num_attrs_idx.shape[0]

        def stat_fn(X):
            rng_h, rng_b = jax.random.split(jax_rng, 2)
            hs_mat = 2 * (jax.random.uniform(rng_h, shape=(num_hs_projections, numeric_dim)) - 0.5) / jnp.sqrt(numeric_dim)  # h x d

            X = X.reshape((-1, dim))
            X_cat_proj = X[:, idx]
            X_num_proj = X[:, num_attrs_idx]  # n x d

            HS_proj = jnp.dot(X_num_proj, hs_mat.T)  # n x h

            X_proj = jnp.concatenate((X_cat_proj, HS_proj), axis=1)

            stat = jnp.histogramdd(X_proj, sizes_jnp)[0].flatten() / X.shape[0]
            return stat

        return stat_fn, np.sqrt(2)

    def get_differentiable_stats_fn_helper(self, kway_attributes):
        assert not self.IS_REAL_VALUED, "Does not work with real-valued data. Must discretize first."
        # For computing differentiable marginal queries
        queries = []
        indices = [self.domain.get_attribute_onehot_indices(att) for att in kway_attributes]
        for tup in itertools.product(*indices):
            queries.append(tup)
        queries = jnp.array(queries)
        # queries_split = jnp.array_split(self.queries, 10)

        # @jax.jit
        def stat_fn(X):
            return jnp.prod(X[:, queries], 2).sum(0) / X.shape[0]

        return stat_fn, jnp.sqrt(2)
    ##################################################
    ## Adaptive statistics
    ##################################################
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
                        self.row_answer_fn[worse_index],
                        self.diff_marginals_fn[worse_index] if not self.IS_REAL_VALUED else None)

        return state

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

            state.add_stats(true_stat, priv_stat, self.marginals_fn[i], self.row_answer_fn[i],
                            self.diff_marginals_fn[i] if not self.IS_REAL_VALUED else None)

        return state



def exponential_mechanism(key:jnp.ndarray, scores: jnp.ndarray, eps0: float, sensitivity: float):
    dist = jax.nn.softmax(2 * eps0 * scores / (2 * sensitivity))
    cumulative_dist = jnp.cumsum(dist)
    max_query_idx = jnp.searchsorted(cumulative_dist, jax.random.uniform(key, shape=(1,)))
    return max_query_idx[0]


######################################################################
## TEST
######################################################################


def test_mixed():

    import numpy as np
    import pandas as pd
    print('debug')
    cols = ['A', 'B', 'C']
    domain = Domain(cols, [2, 1, 1])

    A = pd.DataFrame(np.array([
        [0, 0.0, 0.1],
        [0, 0.2, 0.3],
        [0, 0.8, 0.9],
        [0, 0.1, 0.0],
    ]), columns=cols)
    data = Dataset(A, domain=domain)
    X = data.to_numpy()

    stat_mod = Halfspace.get_all_kway_combinations(domain, 2, random_hs=100)
    stat_mod.fit(data)

    stats1 = stat_mod.get_true_stats()
    print(stats1.shape)
    print(stats1)




def test_hs():
    import numpy as np
    import pandas as pd
    print('debug')
    cols = ['A', 'B', 'C', 'D']
    domain = Domain(cols, [2, 1, 1, 1])

    A = pd.DataFrame(np.array([
        [0, 0.0, 0.1, 0.0],
        [0, 0.2, 0.3, 0.0],
        [0, 0.8, 0.9, 0.0],
        [0, 0.1, 0.0, 0.0],
    ]), columns=cols)
    data = Dataset(A, domain=domain)
    X = data.to_numpy()

    stat_mod = Halfspace(domain, kway_combinations=[('A', 'B', 'C')], num_random_halfspaces=10)
    key = jax.random.PRNGKey(0)
    stat_mod.fit(data, key)

    fn, _ = stat_mod.get_halfspace_marginal_stats_fn_helper(cols=('A', 'B', 'C'), jax_rng=key, num_hs_projections=3)

    stats = fn(X)
    print(stats)

if __name__ == "__main__":
    # test_mixed()
    # test_runtime()
    # test_real_and_diff()
    # test_discrete()
    # test_row_answers()
    test_hs()