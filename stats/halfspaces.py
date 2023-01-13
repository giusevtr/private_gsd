import itertools
import jax
import jax.numpy as jnp
from utils import Dataset, Domain
from utils.utils_data import Domain
from stats import Marginals
import numpy as np
import chex


class Halfspace(Marginals):
    true_stats: list
    marginals_fn: list
    marginals_fn_jit: list
    get_marginals_fn: list
    diff_marginals_fn: list
    diff_marginals_fn_jit: list
    sensitivity: list
    def __init__(self, domain: Domain, kway_combinations: list, rng: chex.PRNGKey, num_random_halfspaces: int = 10):
        """
        :param domain:
        :param kway_combinations:
        :param num_random_halfspaces: number of random halfspaces for each marginal that contains a real-valued feature
        """
        super().__init__(domain, kway_combinations)
        self.num_hs = num_random_halfspaces
        self.rng = rng

    def __str__(self):
        return f'Halfspaces'

    @staticmethod
    def get_kway_random_halfspaces(domain: Domain, k: int, rng:chex.PRNGKey, random_hs: int = 32):
        kway_combinations = []
        for cols in itertools.combinations(domain.attrs, k):
            count_real = 0
            for c in list(cols):
                count_real += 1 * (c in domain.get_numeric_cols())
            if 0 < count_real < k:
                kway_combinations.append(list(cols))
        return Halfspace(domain, kway_combinations, rng=rng, num_random_halfspaces=random_hs), kway_combinations


    def fit(self, data: Dataset):
        self.true_stats = []
        self.marginals_fn = []
        self.marginals_fn_jit = []
        self.diff_marginals_fn = []
        self.diff_marginals_fn_jit = []
        self.get_marginals_fn = []
        self.sensitivity = []

        X = data.to_numpy()
        self.N = X.shape[0]
        self.domain = data.domain

        def get_get_halfspace_func(cols_arg: tuple, rng_arg: chex.PRNGKey):
            def fn(): return self.get_halfspace_marginal_stats_fn_helper(cols_arg, rng_arg, self.num_hs)[0]
            return fn

        def get_get_cat_func(cols_arg):
            return lambda: self.get_categorical_marginal_stats_fn_helper(cols_arg)[0]

        rng = self.rng
        for cols in self.kway_combinations:
            if self.is_workload_numeric(cols):
                rng, rng_sub = jax.random.split(rng)
                get_marginal_fn = get_get_halfspace_func(cols, rng_sub)
                fn, sensitivity = self.get_halfspace_marginal_stats_fn_helper(cols, rng_sub, num_hs_projections=self.num_hs)
                fn_jit = jax.jit(self.get_halfspace_marginal_stats_fn_helper(cols, rng_sub, num_hs_projections=self.num_hs)[0])
            else:
                fn, sensitivity = self.get_categorical_marginal_stats_fn_helper(cols)
                fn_jit = jax.jit(self.get_categorical_marginal_stats_fn_helper(cols)[0])
                get_marginal_fn = get_get_cat_func(cols)

            self.true_stats.append(fn(X))
            self.marginals_fn.append(fn)
            self.marginals_fn_jit.append(fn_jit)
            self.get_marginals_fn.append(get_marginal_fn)
            self.sensitivity.append(sensitivity / self.N)

            # NOTE: Not implemented for halfspaces yet
            diff_fn, sensitivity = self.get_differentiable_stats_fn_helper(cols)
            diff_fn_jit = jax.jit(self.get_differentiable_stats_fn_helper(cols)[0])
            self.diff_marginals_fn.append(diff_fn)
            self.diff_marginals_fn_jit.append(diff_fn_jit)

    # @jax.jit
    # def get_halfspace
    def get_halfspace_marginal_stats_fn_helper(self, cols: tuple, jax_rng: chex.PRNGKey, num_hs_projections):
        # Get categorical columns meta data
        dim = len(self.domain.attrs)

        cat_cols, numeric_cols = [], []
        for col in cols:
            if col in self.domain.get_categorical_cols():
                cat_cols.append(col)
            else:
                numeric_cols.append(col)

        cat_idx = self.domain.get_attribute_indices(cat_cols)
        sizes = []
        for col in cat_cols:
            sizes.append(self.domain.size(col))
        sizes_jnp = [jnp.arange(s + 1).astype(float) for i, s in zip(cat_idx, sizes)]
        # for i in range(num_hs_projections):
        #     sizes_jnp.append(jnp.array([-1, 0.0, 1]))

        cat_queries = []
        indices = [self.domain.get_attribute_onehot_indices(att) for att in cat_cols]
        for tup in itertools.product(*indices):
            cat_queries.append(tup)
        cat_queries = jnp.array(cat_queries)


        num_idx = self.domain.get_attribute_indices(numeric_cols)
        numeric_dim = num_idx.shape[0]

        def histogramdd_row_level(row_data):
            return jnp.histogramdd(row_data, sizes_jnp)[0].flatten()
        histogramdd_vmap = jax.vmap(histogramdd_row_level, in_axes=(0, ))

        def stat_fn(X):
            n, d = X.shape

            # HS
            rng_h, rng_b = jax.random.split(jax_rng, 2)
            hs_mat = numeric_dim * 2 * (jax.random.uniform(rng_h, shape=(numeric_dim, num_hs_projections)) - 0.5)   # d x h
            b =  2 * (jax.random.uniform(rng_b, shape=(1, num_hs_projections)) - 0.5)  # 1 x h
            X = X.reshape((-1, dim))
            X_num_proj = X[:, num_idx]  # n x d
            HS_proj = jnp.dot(X_num_proj, hs_mat)  # n x h
            above_halfspace = HS_proj - b > 0  # n x 1

            # Cat
            X_cat_proj = X[:, cat_idx].reshape((n, 1, -1))
            cat_answers = histogramdd_vmap(X_cat_proj)
            # cat_answers = jnp.prod(X[:, cat_queries], 2)   # n x m

            answers = jnp.multiply(cat_answers.reshape((n, -1, 1)), above_halfspace.reshape((n, 1, -1))).reshape((n, -1))
            statistics = answers.sum(0) / X.shape[0]
            return statistics

        return stat_fn, np.sqrt(2)

    def get_differentiable_stats_fn_helper(self, kway_attributes):
        # assert not self.IS_REAL_VALUED, "Does not work with real-valued data. Must discretize first."
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