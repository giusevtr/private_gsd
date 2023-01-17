import itertools
import jax
import jax.numpy as jnp
from utils import Dataset, Domain
from utils.utils_data import Domain
from stats import Marginals
import numpy as np
import chex


class MarginalsPrefix(Marginals):
    true_stats: list
    marginals_fn: list
    marginals_fn_jit: list
    get_marginals_fn: list
    get_differentiable_fn: list
    diff_marginals_fn: list
    diff_marginals_fn_jit: list
    sensitivity: list
    def __init__(self, domain: Domain, kway_combinations: list, rng: chex.PRNGKey,
                 prefix_workloads: int,
                 num_random_prefices: int):
        """
        :param domain:
        :param kway_combinations:
        :param num_random_prefices: number of random halfspaces for each marginal that contains a real-valued feature
        """
        super().__init__(domain, kway_combinations)
        self.num_prefix_workloads = prefix_workloads
        self.num_prefix_samples = num_random_prefices
        self.rng = rng

    def __str__(self):
        return f'Halfspaces'

    @staticmethod
    def get_kway_random_prefices(domain: Domain, k: int, rng:chex.PRNGKey,
                                 prefix_workloads: int = 1,
                                 random_prefices: int = 500):
        kway_combinations = []
        for cols in itertools.combinations(domain.attrs, k):
            count_real = 0
            for c in list(cols):
                count_real += 1 * (c in domain.get_numeric_cols())
            if 0 < count_real < k:
                # For now the workload must contain at least 1 categorical and 1 numerical
                kway_combinations.append(list(cols))
        return MarginalsPrefix(domain, kway_combinations, rng=rng, prefix_workloads=prefix_workloads,
                         num_random_prefices=random_prefices), kway_combinations


    def fit(self, data: Dataset):
        self.true_stats = []
        self.marginals_fn = []
        self.marginals_fn_jit = []
        self.diff_marginals_fn = []
        self.diff_marginals_fn_jit = []
        self.get_marginals_fn = []
        self.get_differentiable_fn = []
        self.sensitivity = []

        X = data.to_numpy()
        self.N = X.shape[0]
        self.domain = data.domain

        def get_get_halfspace_func(cols_arg: tuple, rng_arg: chex.PRNGKey):
            return lambda: self.get_prefix_stats_fn_helper(cols_arg, rng_arg, self.num_prefix_samples)[0]
        def get_get_differentiable_halfspace_func(cols_arg: tuple, rng_arg: chex.PRNGKey):
            return lambda: self.get_diff_prefix_fn_helper(cols_arg, rng_arg, self.num_prefix_samples)[0]

        def get_get_cat_func(cols_arg):
            return lambda: self.get_categorical_marginal_stats_fn_helper(cols_arg)[0]

        rng = self.rng
        for cols in self.kway_combinations:
            if self.is_workload_numeric(cols):
                for _ in range(self.num_prefix_workloads):
                    rng, rng_sub = jax.random.split(rng)
                    fn, sensitivity = self.get_prefix_stats_fn_helper(cols, rng_sub, num_prefices=self.num_prefix_samples)
                    get_marginal_fn = get_get_halfspace_func(cols, rng_sub)
                    get_differentiable_fn = get_get_differentiable_halfspace_func(cols, rng_sub)
                    fn_jit = jax.jit(get_marginal_fn())

                    self.true_stats.append(fn(X))
                    self.marginals_fn.append(fn)
                    self.marginals_fn_jit.append(fn_jit)
                    self.get_marginals_fn.append(get_marginal_fn)
                    self.sensitivity.append(sensitivity / self.N)

                    diff_fn, sensitivity = self.get_diff_prefix_fn_helper(cols, rng_sub, num_prefices=self.num_prefix_samples)
                    diff_fn_jit = jax.jit(get_differentiable_fn())
                    self.diff_marginals_fn.append(diff_fn)
                    self.diff_marginals_fn_jit.append(diff_fn_jit)
                    self.get_differentiable_fn.append(get_differentiable_fn)

            else:
                fn, sensitivity = self.get_categorical_marginal_stats_fn_helper(cols)
                fn_jit = jax.jit(self.get_categorical_marginal_stats_fn_helper(cols)[0])
                get_marginal_fn = get_get_cat_func(cols)

                self.true_stats.append(fn(X))
                self.marginals_fn.append(fn)
                self.marginals_fn_jit.append(fn_jit)
                self.get_marginals_fn.append(get_marginal_fn)
                self.sensitivity.append(sensitivity / self.N)

                diff_fn, sensitivity = self.get_differentiable_stats_fn_helper(cols)
                diff_fn_jit = jax.jit(self.get_differentiable_stats_fn_helper(cols)[0])
                self.diff_marginals_fn.append(diff_fn)
                self.diff_marginals_fn_jit.append(diff_fn_jit)

    # @jax.jit
    # def get_halfspace
    def get_prefix_stats_fn_helper(self, cols: tuple, jax_rng: chex.PRNGKey, num_prefices):
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
        cat_idx = jnp.concatenate((cat_idx, jnp.array([dim]).astype(int))).astype(int)
        sizes.append(1)
        sizes_jnp = [jnp.arange(s + 1).astype(float) for i, s in zip(cat_idx, sizes)]


        num_idx = self.domain.get_attribute_indices(numeric_cols)
        numeric_dim = num_idx.shape[0]

        def histogramdd_row_level(row_data):
            return jnp.histogramdd(row_data, sizes_jnp)[0].flatten()
        histogramdd_vmap = jax.vmap(histogramdd_row_level, in_axes=(0, ))

        def stat_fn(X):
            n, d = X.shape
            X = jnp.column_stack((X, jnp.ones(n).astype(int)))

            # Cat
            X_cat_proj = X[:, cat_idx].reshape((n, 1, -1))
            cat_answers = histogramdd_vmap(X_cat_proj)

            # Prefix
            prefix_thresholds = jax.random.uniform(key=jax_rng, shape=(1, numeric_dim,  num_prefices))  # d_num x q
            X_num_proj = X[:, num_idx].reshape((n, numeric_dim))  # n x d_num
            X_num_proj_tile = jnp.tile(X_num_proj, num_prefices).reshape((n, numeric_dim, num_prefices))
            answers = X_num_proj_tile < prefix_thresholds  # n x d_num x q
            num_answers = jnp.prod(answers, axis=1)  # n x q

            answers = jnp.multiply(cat_answers.reshape((n, -1, 1)), num_answers.reshape((n, 1, -1))).reshape((n, -1))
            statistics = answers.sum(0) / X.shape[0]
            return statistics

        return stat_fn, np.sqrt(num_prefices)

    def get_diff_prefix_fn_helper(self, cols, jax_rng: chex.PRNGKey, num_prefices: int):
        cat_cols, numeric_cols = [], []
        for col in cols:
            if col in self.domain.get_categorical_cols():
                cat_cols.append(col)
            else:
                numeric_cols.append(col)

        # assert not self.IS_REAL_VALUED, "Does not work with real-valued data. Must discretize first."
        # For computing differentiable marginal queries
        cat_queries = []
        indices = [self.domain.get_attribute_onehot_indices(att) for att in cat_cols]
        for tup in itertools.product(*indices):
            cat_queries.append(tup)
        cat_queries = jnp.array(cat_queries).astype(int)
        # queries_split = jnp.array_split(self.queries, 10)

        # num_idx = self.domain.get_attribute_onehot_indices(numeric_cols)
        num_idx = jnp.array([self.domain.get_attribute_onehot_indices(att) for att in numeric_cols]).flatten()

        numeric_dim = num_idx.shape[0]
        def stat_fn(X, sigmoid):
            n, d = X.shape

            # Categorical Marginals
            cat_answers = jnp.prod(X[:, cat_queries], 2)

            # Prefices
            prefix_thresholds = jax.random.uniform(key=jax_rng, shape=(1, numeric_dim,  num_prefices))  # d_num x q
            X_num_proj = X[:, num_idx].reshape((n, numeric_dim))  # n x d_num
            X_num_proj_tile = jnp.tile(X_num_proj, num_prefices).reshape((n, numeric_dim, num_prefices))
            answers = jax.nn.sigmoid(-sigmoid * (X_num_proj_tile - prefix_thresholds))  # n x d_num x q
            num_answers = jnp.prod(answers, axis=1)  # n x q

            diff_answers = jnp.multiply(cat_answers.reshape((n, -1, 1)), num_answers.reshape((n, 1, -1))).reshape((n, -1))
            diff_statistics = diff_answers.sum(0) / X.shape[0]

            return diff_statistics

        return stat_fn, jnp.sqrt(num_prefices)



######################################################################
## TEST
######################################################################



def test_prefix():
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

    cols = ('B', 'C')
    key = jax.random.PRNGKey(0)
    stat_mod = MarginalsPrefix(domain, kway_combinations=[cols], rng=key, prefix_workloads=2, num_random_prefices=5)
    stat_mod.fit(data)

    print(stat_mod.true_stats)

    fn = stat_mod.marginals_fn[0]
    stats = stat_mod.true_stats[0]
    # # stats = fn(data.to_numpy())
    #
    fn_diff = stat_mod.diff_marginals_fn[0]
    stats_diff = fn_diff(data.to_onehot(), sigmoid=1000)
    print(stats)
    print(stats_diff)
    print(stats - stats_diff)

if __name__ == "__main__":
    # test_mixed()
    # test_runtime()
    # test_real_and_diff()
    # test_discrete()
    # test_row_answers()
    test_prefix()