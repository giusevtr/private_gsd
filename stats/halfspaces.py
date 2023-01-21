import itertools
import jax
import jax.numpy as jnp
from utils import Dataset, Domain
from utils.utils_data import Domain
from stats import Marginals
import numpy as np
import chex
import time

class Halfspace(Marginals):
    true_stats: list
    marginals_fn: list
    marginals_fn_jit: list
    get_marginals_fn: list
    get_differentiable_fn: list
    diff_marginals_fn: list
    diff_marginals_fn_jit: list
    sensitivity: list
    def __init__(self, domain: Domain, kway_combinations: list, rng: chex.PRNGKey,
                 num_random_halfspaces: int):
        """
        :param domain:
        :param kway_combinations:
        :param num_random_halfspaces: number of random halfspaces for each marginal that contains a real-valued feature
        """
        super().__init__(domain, kway_combinations)
        self.num_hs_samples = num_random_halfspaces
        self.rng = rng

    def __str__(self):
        return f'Halfspaces'

    @staticmethod
    def get_kway_random_halfspaces(domain: Domain,
                                   k: int,
                                   rng:chex.PRNGKey,
                                   random_hs: int = 500):
        kway_combinations = []
        for cols in itertools.combinations(domain.get_categorical_cols(), k):
            kway_combinations.append(list(cols))
        return Halfspace(domain, kway_combinations, rng=rng,
                         num_random_halfspaces=random_hs), kway_combinations


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
        dim = len(self.domain.attrs)

        # def get_get_halfspace_func(cols_arg: tuple, rng_arg: chex.PRNGKey):
        #     return lambda: self.get_halfspace_marginal_stats_fn_helper(cols_arg, rng_arg, self.num_hs_samples)[0]
        # def get_get_differentiable_halfspace_func(cols_arg: tuple, rng_arg: chex.PRNGKey):
        #     return lambda: self.get_diff_halfspace_fn_helper(cols_arg, rng_arg, self.num_hs_samples)[0]
        # def get_get_cat_func(cols_arg):
        #     return lambda: self.get_categorical_marginal_stats_fn_helper(cols_arg)[0]

        # self.marginal_idx = []
        # for cols in self.kway_combinations:
        #     self.marginal_idx.append(self.domain.get_attribute_indices(cols))
        # self.marginal_idx.append((dim, ))

        self.halfspace_keys = jax.random.split(self.rng, self.num_hs_samples)
        self.halfpaces_stats = []
        self.halfpaces_fn = []
        self.halfpaces_fn_jit = []
        self.halfpaces_fn_diff_jit = []

        # self.kway_combinations = self.kway_combinations + [[]]

        self.halfspace_map = {}
        query_id = 0
        for marginal_id, cols in enumerate(self.kway_combinations):
            halfspace_fn_vmap = self.get_halfspace_stats_fn_helper(cols, self.halfspace_keys)
            diff_halfspace_fn_vmap = self.get_diff_halfspace_fn_helper(cols, self.halfspace_keys)


            # Compute stats on orginal data
            X = data.to_numpy()
            total_num_rows = X.shape[0]
            if total_num_rows <= 2000:
                hs_stats = halfspace_fn_vmap(X)
            else:
                num_splits = total_num_rows // 2000
                X_split = jnp.array_split(data.to_numpy(), num_splits)
                stat_sum = None
                for i in range(num_splits):
                    X_i = X_split[i]
                    num_rows = X_i.shape[0]
                    temp_stats = num_rows * halfspace_fn_vmap(X_i)
                    stat_sum = temp_stats if stat_sum is None else stat_sum + temp_stats
                hs_stats = stat_sum / total_num_rows

            self.halfpaces_stats.append(hs_stats)
            self.halfpaces_fn.append(halfspace_fn_vmap)
            self.halfpaces_fn_jit.append(jax.jit(halfspace_fn_vmap))
            self.halfpaces_fn_diff_jit.append(jax.jit(diff_halfspace_fn_vmap,))

            for hs_sample_id in range(self.num_hs_samples):
                self.halfspace_map[query_id] = (marginal_id, hs_sample_id)
                query_id = query_id + 1
                self.sensitivity.append(jnp.sqrt(2) / self.N)

    def get_num_queries(self):
        return len(self.sensitivity)

    def get_true_stat(self, stat_ids: list):
        stats = []
        for i in stat_ids:
            i = int(i)
            marginal_id, hs_sample_id = self.halfspace_map[i]
            temp_hs_stats = self.halfpaces_stats[marginal_id].reshape((self.num_hs_samples, -1))[hs_sample_id]
            stats.append(temp_hs_stats)
        return jnp.concatenate(stats)

    def get_stat_fn(self, stat_ids: list):

        # stat_position = {}
        stat_fn_list = []
        hs_keys_maps = {}
        for pos, stat_id in enumerate(stat_ids):
            stat_id = int(stat_id)
            marginal_id, hs_sample_id = self.halfspace_map[stat_id]
            if marginal_id not in hs_keys_maps:
                hs_keys_maps[marginal_id] = []
            hs_keys_maps[marginal_id].append(hs_sample_id)

        for marginal_id in hs_keys_maps:
            cols = self.kway_combinations[marginal_id]
            hs_keys_ids = jnp.array(hs_keys_maps[marginal_id])
            hs_keys = self.halfspace_keys[hs_keys_ids].reshape(hs_keys_ids.shape[0], -1)
            stat_fn_list.append(self.get_halfspace_stats_fn_helper(cols, hs_keys))
            # stat_fn_list[st]

        def stat_fn(X):
            stats = jnp.concatenate([fn(X) for fn in stat_fn_list])
            return stats
        return stat_fn

    def get_diff_stat_fn(self, stat_ids: list):
        hs_keys_maps = {}
        for i in stat_ids:
            i = int(i)
            marginal_id, hs_sample_id = self.halfspace_map[i]
            if marginal_id not in hs_keys_maps:
                hs_keys_maps[marginal_id] = []
            hs_keys_maps[marginal_id].append(hs_sample_id)

        stat_fn_list = []
        for marginal_id in hs_keys_maps:
            cols = self.kway_combinations[marginal_id]
            hs_keys_ids = jnp.array(hs_keys_maps[marginal_id])
            hs_keys = self.halfspace_keys[hs_keys_ids].reshape(hs_keys_ids.shape[0], -1)
            stat_fn_list.append(self.get_diff_halfspace_fn_helper(cols, hs_keys))

        def stat_fn(X, sigmoid):
            stats = jnp.concatenate([fn(X, sigmoid) for fn in stat_fn_list])
            return stats

        return stat_fn

    def get_sync_data_errors(self, X):
        assert self.true_stats is not None, "Error: must call the fit function"
        max_errors = []
        for i in range(len(self.halfpaces_stats)):
            fn_jit = self.halfpaces_fn_jit[i]
            sync_stat = fn_jit(X).reshape((self.num_hs_samples, -1))
            true_stat = self.halfpaces_stats[i].reshape((self.num_hs_samples, -1))
            max_error = jnp.abs(true_stat - sync_stat).max(axis=1)
            max_errors.append(max_error)
        max_errors_jax = jnp.concatenate(max_errors)
        return max_errors_jax


    # @jax.jit
    # def get_halfspace
    def get_halfspace_stats_fn_helper(self, cols: tuple, keys: chex.PRNGKey):
        # Get categorical columns meta data
        dim = len(self.domain.attrs)

        num_hs_queries = keys.shape[0]

        cat_cols = []
        for col in cols:
            if col in self.domain.get_categorical_cols():
                cat_cols.append(col)

        cat_idx = self.domain.get_attribute_indices(cat_cols)
        sizes = []
        for col in cat_cols:
            sizes.append(self.domain.size(col))
        cat_idx = jnp.concatenate((cat_idx, jnp.array([dim]).astype(int))).astype(int)
        sizes.append(1)
        sizes_jnp = [jnp.arange(s + 1).astype(float) for i, s in zip(cat_idx, sizes)]

        numeric_cols = self.domain.get_numeric_cols()
        num_idx = self.domain.get_attribute_indices(numeric_cols)
        numeric_dim = num_idx.shape[0]

        # @jax.jit
        def stat_fn(x_row, key):
            # Compute statistic for a single row and halfspace
            n=1
            x_row = jnp.concatenate((x_row, jnp.ones(n).astype(int)))

            # Cat
            x_row_cat_proj = x_row[cat_idx].reshape((1, -1))
            cat_answers = jnp.histogramdd(x_row_cat_proj, sizes_jnp)[0].flatten()

            # HS
            rng_h, rng_b = jax.random.split(key, 2)
            hs_mat = (jax.random.normal(rng_h, shape=(numeric_dim,))) / jnp.sqrt(numeric_dim)  # d x h
            b = jax.random.normal(rng_b, shape=(1,))  # 1 x h
            x_row_num_proj = x_row[num_idx]  # n x d
            HS_proj = jnp.dot(x_row_num_proj, hs_mat) - b  # n x h
            above_halfspace = (HS_proj > 0).astype(int)  # n x h

            answers = cat_answers * above_halfspace
            return answers

        row_stat_fn_vmap = jax.vmap(stat_fn, in_axes=(None, 0))  # iterates over halfspaces
        stat_fn_vmap = jax.vmap(row_stat_fn_vmap, in_axes=(0, None))  # Iterates over rows

        if num_hs_queries == 1:
            def stat_fn_vmap2(X):
                answers = stat_fn_vmap(X, keys)
                return (answers.sum(0) / X.shape[0]).reshape(-1)
        else:
            def stat_fn_vmap2(X):
                answers = stat_fn_vmap(X, keys)
                answers = answers.reshape((X.shape[0], -1))
                return (answers.sum(0) / X.shape[0])

        return stat_fn_vmap2

    def get_diff_halfspace_fn_helper(self, cols, keys: chex.PRNGKey):
        cat_cols = []
        for col in cols:
            if col in self.domain.get_categorical_cols():
                cat_cols.append(col)
        # For computing differentiable marginal queries
        cat_queries = []
        indices = [self.domain.get_attribute_onehot_indices(att) for att in cat_cols] + [jnp.array([-1])]
        for tup in itertools.product(*indices):
            cat_queries.append(tup)
        cat_queries = jnp.array(cat_queries)
        numeric_cols = self.domain.get_numeric_cols()
        num_idx = jnp.array([self.domain.get_attribute_onehot_indices(att) for att in numeric_cols]).flatten()
        numeric_dim = num_idx.shape[0]

        def stat_fn(X, sigmoid, key):
            n, d = X.shape
            X = jnp.column_stack((X, jnp.ones(n).astype(int)))
            cat_answers = jnp.prod(X[:, cat_queries], 2)
            # Compute halfspace answers
            rng_h, rng_b = jax.random.split(key, 2)
            hs_mat = (jax.random.normal(rng_h, shape=(numeric_dim,))) / jnp.sqrt(numeric_dim)  # d x h
            b = jax.random.normal(rng_b, shape=(1,))  # 1 x h
            X_num_proj = X[:, num_idx]  # n x d
            HS_proj = jnp.dot(X_num_proj, hs_mat) - b # n x h
            above_halfspace = jax.nn.sigmoid(sigmoid * HS_proj) # n x h
            diff_answers = jnp.multiply(cat_answers.reshape((n, -1, 1)), above_halfspace.reshape((n, 1, -1))).reshape((n, -1))
            diff_statistics = diff_answers.sum(0) / X.shape[0]

            return diff_statistics
        num_hs_queries = keys.shape[0]

        stat_fn_vmap = jax.vmap(stat_fn, in_axes=(None, None, 0))
        # stat_fn_vmap2 = lambda X, sigmoid: stat_fn_vmap(X, sigmoid, keys).reshape((num_hs_queries, -1))
        if num_hs_queries == 1:
            stat_fn_vmap2 = lambda X, sigmoid: stat_fn_vmap(X, sigmoid, keys).reshape((-1))
        else:
            stat_fn_vmap2 = lambda X, sigmoid: stat_fn_vmap(X, sigmoid, keys).reshape(-1)
        return stat_fn_vmap2

    def get_true_stats(self):
        assert self.true_stats is not None, "Error: must call the fit function"
        return jnp.concatenate([stat.flatten() for stat in self.halfpaces_stats])

    def get_stats(self, data: Dataset, indices: list = None):
        X = data.to_numpy()
        stats = self.get_stats_jax(X)
        return stats

    def get_stats_jit(self, data: Dataset, indices: list = None):
        X = data.to_numpy()
        stats = self.get_stats_jax_jit(X)
        return stats

    def get_stats_jax(self, X: chex.Array):
        stats = jnp.concatenate([fn(X).flatten() for fn in self.halfpaces_fn])
        return stats

    def get_stats_jax_jit(self, X: chex.Array):
        stats = jnp.concatenate([fn(X).flatten() for fn in self.halfpaces_fn_jit])
        return stats

    def get_diff_halfspace_stats(self, data: Dataset, sigmoid, indices: list = None):
        X = data.to_onehot()
        stats = []
        I = indices if indices is not None else list(range(self.get_num_queries()))
        for stat_fn_diff_jit in self.halfpaces_fn_diff_jit:
            stats.append(stat_fn_diff_jit(X, sigmoid).flatten())
        return jnp.concatenate(stats)



######################################################################
## TEST
######################################################################
import pandas as pd


def test_get_max_errors():
    print('test_get_max_errors()')
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

    cols = ('A', )
    key = jax.random.PRNGKey(0)
    stat_mod = Halfspace(domain, kway_combinations=[cols], rng=key, num_random_halfspaces=3)
    stat_mod.fit(data)


def test_hs():

    print('debug')
    cols = ['A', 'B', 'C', 'D', 'E', 'F']
    domain = Domain(cols, [2, 5, 2, 1, 1, 1])

    A = pd.DataFrame(np.array([
        [0, 0, 1, 0.0, 0.1, 0.0],
        [0, 2, 1, 0.2, 0.3, 0.0],
        [0, 3, 1, 0.8, 0.9, 0.0],
        [0, 1, 1, 0.1, 0.0, 0.0],
        # [0, 1, 0.1, 0.0, 0.1],
    ]), columns=cols)
    data = Dataset(A, domain=domain)

    rand_data = Dataset.synthetic(domain, N=10, seed=0)

    cols = [('A',), ('B',), ('C', )]

    num_random_halfspaces = 13
    workloads = len(cols) * num_random_halfspaces + num_random_halfspaces
    # cols = []
    key = jax.random.PRNGKey(0)
    stat_mod = Halfspace(domain, kway_combinations=cols, rng=key, num_random_halfspaces=num_random_halfspaces)
    stat_mod.fit(data)

    stat_fn = stat_mod.get_stat_fn(list(jnp.arange(stat_mod.get_num_queries())))
    diff_stat_fn = stat_mod.get_diff_stat_fn(list(jnp.arange(stat_mod.get_num_queries())))
    # print(stat_mod.get_true_stats())
    # print(stat_fn(data.to_numpy()))

    # print(f'diff stats:')
    # print(diff_stat_fn(data.to_onehot(), 1000))
    # print('diff error = ', jnp.abs(stat_fn(data.to_numpy()) - diff_stat_fn(data.to_onehot(), 1000)).max())
    #
    # errors = stat_mod.get_sync_data_errors(data.to_numpy())
    # assert errors.shape[0] == workloads, f'errors.shape[0] = {errors.shape[0]}, workloads = {workloads}'
    # print(f'max error = {errors.max()}')


    print(f'num queries={stat_mod.get_num_queries()}')

    for qid in range(stat_mod.get_num_queries()):
        stat_fn = stat_mod.get_stat_fn([qid])
        stat = stat_mod.get_true_stat([qid])
        diff = jnp.abs(stat - stat_fn(data.to_numpy()))
        assert diff.max() < 0.01

    np.random.seed(1)
    temp = np.arange(stat_mod.get_num_queries())
    np.random.shuffle(temp)
    ids = list(temp)
    # print(ids)
    for i in range(1, 4):
        temp = ids[:i]
        temp.sort()
        print(temp)
        stat_fn1 = stat_mod.get_stat_fn(temp)
        print(stat_fn1(data.to_numpy()), f'\t\t\t\t ids[:{i:<3}]={temp}:')
        print()

def test_fit_runtime():

    print('debug')
    cols_names = [f'f{i}' for i in range(20)]
    cols_sizes = [3 for _ in range(10)] + [1 for _ in range(10)]
    domain = Domain(cols_names, cols_sizes)
    data = Dataset.synthetic(domain, N=10007, seed=0)
    rand_data = Dataset.synthetic(domain, N=1000, seed=0)

    t0 = time.time()
    key = jax.random.PRNGKey(0)
    stat_mod, _ = Halfspace.get_kway_random_halfspaces(domain, k=1, rng=key, random_hs=10003)
    stat_mod.fit(data)
    t1 = time.time()
    print(f'fit time = {t1 - t0:.5f}')
    stat_mod.get_true_stats()
    temp = stat_mod.get_stats_jit(rand_data)
    print(f'\tstat_size = {temp.shape}')
    t2 = time.time()
    print(f'get_stat_jit time = {t2 - t1:.5f}')
    stat_mod.get_sync_data_errors(rand_data.to_numpy())
    t3 = time.time()
    print(f'get_sync_data_errors time = {t3 - t2:.5f}')


if __name__ == "__main__":
    # test_mixed()
    # test_runtime()
    # test_real_and_diff()
    # test_discrete()
    # test_row_answers()
    test_hs()
    # test_fit_runtime()