import itertools
import jax
import jax.numpy as jnp
from utils import Dataset, Domain
from utils.utils_data import Domain
from stats import Marginals
import numpy as np
import chex
import time
from stats import AdaptiveStatisticState

from tqdm import tqdm

class Prefix(AdaptiveStatisticState):

    def __init__(self, domain: Domain,
                 k_cat: int,
                 cat_kway_combinations: list,
                 rng: chex.PRNGKey,
                 k_prefix: int,
                 num_random_prefixes: int):
        """
        :param domain:
        :param cat_kway_combinations:
        :param num_random_prefixes: number of random halfspaces for each marginal that contains a real-valued feature
        """
        # super().__init__(domain, kway_combinations)
        self.domain = domain
        self.cat_kway_combinations = cat_kway_combinations
        self.num_prefix_samples = num_random_prefixes
        self.k = k_cat
        self.k_prefix = k_prefix
        self.rng = rng

        self.prefix_keys = jax.random.split(self.rng, self.num_prefix_samples)
        self.workload_positions = []
        self.workload_sensitivity = []

        self.set_up_stats()

    def get_num_workloads(self):
        return len(self.workload_positions)

    def _get_workload_positions(self, workload_id: int = None) -> tuple:
        return self.workload_positions[workload_id]

    def _get_workload_sensitivity(self, workload_id: int = None, N: int = None) -> float:
        return self.workload_sensitivity[workload_id] / N

    def set_up_stats(self):

        queries = []
        self.workload_positions = []
        for marginal in tqdm(self.cat_kway_combinations, desc='Setting up Prefix.'):
            assert len(marginal) == self.k
            indices = self.domain.get_attribute_indices(marginal)

            for prefix_pos in range(self.num_prefix_samples):
                start_pos = len(queries)
                intervals = []
                for att in marginal:
                    size = self.domain.size(att)
                    assert size>1
                    upper = np.linspace(0, size, num=size+1)[1:]
                    lower = np.linspace(0, size, num=size+1)[:-1]
                    # lower = lower.at[0].set(-0.01)

                    intervals_arr = np.vstack((upper, lower)).T - 0.1
                    interval_list = list(intervals_arr)
                    intervals.append(interval_list)

                for v in itertools.product(*intervals):
                    v_arr = np.array(v)
                    upper = v_arr.flatten()[::2]
                    lower = v_arr.flatten()[1::2]
                    q = np.concatenate((indices, upper, lower, np.array([prefix_pos])))
                    queries.append(q)
                end_pos = len(queries)
                self.workload_positions.append((start_pos, end_pos))
                self.workload_sensitivity.append(jnp.sqrt(2))

        self.queries = jnp.array(queries)

# (i1, i2), ((a1, a2), (b1, b2))
    def _get_workload_fn(self, workload_ids=None):
        """
        Returns marginals function and sensitivity
        :return:
        """
        dim = len(self.domain.attrs)
        numeric_cols = self.domain.get_numeric_cols()
        num_idx = self.domain.get_attribute_indices(numeric_cols)
        numeric_dim = num_idx.shape[0]


        def answer_fn(x_row: chex.Array, query_single: chex.Array):
            I = query_single[:self.k].astype(int)
            U = query_single[self.k:2*self.k]
            L = query_single[2*self.k:3*self.k]
            key_id = query_single[-1].astype(int)
            prefix_key = self.prefix_keys[key_id]

            # Categorical
            t1 = (x_row[I] < U).astype(int)
            t2 = (x_row[I] >= L).astype(int)
            t3 = jnp.prod(jnp.array([t1, t2]), axis=0)
            cat_answers = jnp.prod(t3)

            # Prefix
            rng_h, rng_b = jax.random.split(prefix_key, 2)
            thresholds = jax.random.uniform(rng_h, shape=(self.k_prefix,)) # d x h
            pos = jax.random.randint(rng_b, minval=0, maxval=numeric_dim, shape=(self.k_prefix,)) # d x h
            kway_idx = num_idx[pos]
            below_threshold = (x_row[kway_idx] < thresholds).astype(int)  # n x d
            prefix_answer = jnp.prod(below_threshold)

            answers = cat_answers * prefix_answer

            return answers

        if workload_ids is None :
            these_queries = self.queries
        else:
            these_queries = []
            query_positions = []
            for stat_id in workload_ids:
                a, b = self.workload_positions[stat_id]
                q_pos = jnp.arange(a, b)
                query_positions.append(q_pos)
                these_queries.append(self.queries[a:b, :])
            these_queries = jnp.concatenate(these_queries, axis=0)

        temp_rows_fn = jax.vmap(answer_fn, in_axes=(None, 0))
        temp_stat_fn = jax.jit(jax.vmap(temp_rows_fn, in_axes=(0, None)))

        def stat_fn(X):
            return temp_stat_fn(X, these_queries).sum(0) / X.shape[0]

        return stat_fn

    @staticmethod
    def get_kway_prefixes(domain: Domain,
                          k_cat: int,
                          k_num: int,
                          rng: chex.PRNGKey,
                          random_prefixes: int = 500):
        cat_kway_combinations = []
        for cols in itertools.combinations(domain.get_categorical_cols(), k_cat):
            cat_kway_combinations.append(list(cols))
        return Prefix(domain, k_cat=k_cat, cat_kway_combinations=cat_kway_combinations, rng=rng,
                      k_prefix=k_num,
                      num_random_prefixes=random_prefixes)


#     def __str__(self):
#         return f'Prefix'
#
#     @staticmethod
#     def get_kway_prefixes(domain: Domain,
#                           k: int,
#                           rng:chex.PRNGKey,
#                           random_prefixes: int = 500):
#         kway_combinations = []
#         for cols in itertools.combinations(domain.get_categorical_cols(), k):
#             kway_combinations.append(list(cols))
#         return Prefix(domain, kway_combinations, rng=rng,
#                       num_random_prefixes=random_prefixes), kway_combinations
#
#
#     def fit(self, data: Dataset):
#         self.true_stats = []
#         self.marginals_fn = []
#         self.marginals_fn_jit = []
#         self.diff_marginals_fn = []
#         self.diff_marginals_fn_jit = []
#         self.get_marginals_fn = []
#         self.get_differentiable_fn = []
#         self.sensitivity = []
#
#         X = data.to_numpy()
#         self.N = X.shape[0]
#         self.domain = data.domain
#         dim = len(self.domain.attrs)
#
#
#         self.prefix_keys = jax.random.split(self.rng, self.num_prefix_samples)
#         self.prefix_stats = []
#         self.prefix_fn = []
#         self.prefix_fn_jit = []
#         self.prefix_fn_diff_jit = []
#
#         # self.kway_combinations = self.kway_combinations + [[]]
#
#         self.prefix_map = {}
#         query_id = 0
#         self.kway_combinations.append('sentinel')
#
#         for marginal_id, cols in enumerate(self.kway_combinations):
#             prefix_fn_vmap = self.get_prefix_fn_helper(cols, self.prefix_keys)
#             diff_prefix_fn_vmap = self.get_diff_prefix_fn_helper(cols, self.prefix_keys)
#
#             # Compute stats on orginal data
#             X = data.to_numpy()
#             total_num_rows = X.shape[0]
#             if total_num_rows <= 2000:
#                 hs_stats = prefix_fn_vmap(X)
#             else:
#                 num_splits = total_num_rows // 2000
#                 X_split = jnp.array_split(data.to_numpy(), num_splits)
#                 stat_sum = None
#                 for i in range(num_splits):
#                     X_i = X_split[i]
#                     num_rows = X_i.shape[0]
#                     temp_stats = num_rows * prefix_fn_vmap(X_i)
#                     stat_sum = temp_stats if stat_sum is None else stat_sum + temp_stats
#                 hs_stats = stat_sum / total_num_rows
#
#             self.prefix_stats.append(hs_stats)
#             self.prefix_fn.append(prefix_fn_vmap)
#             self.prefix_fn_jit.append(jax.jit(prefix_fn_vmap))
#             self.prefix_fn_diff_jit.append(jax.jit(diff_prefix_fn_vmap, ))
#
#             for hs_sample_id in range(self.num_prefix_samples):
#                 self.prefix_map[query_id] = (marginal_id, hs_sample_id)
#                 query_id = query_id + 1
#                 self.sensitivity.append(jnp.sqrt(2) / self.N)
#
#     def get_num_workloads(self):
#         return len(self.sensitivity)
#
#     def get_true_stat(self, stat_ids: list):
#         stats = []
#         for i in stat_ids:
#             i = int(i)
#             marginal_id, hs_sample_id = self.prefix_map[i]
#             temp_hs_stats = self.prefix_stats[marginal_id].reshape((self.num_prefix_samples, -1))[hs_sample_id]
#             stats.append(temp_hs_stats)
#         return jnp.concatenate(stats)
#
#     def get_stat_fn(self, stat_ids: list):
#
#         # stat_position = {}
#         stat_fn_list = []
#         hs_keys_maps = {}
#         for pos, stat_id in enumerate(stat_ids):
#             stat_id = int(stat_id)
#             marginal_id, hs_sample_id = self.prefix_map[stat_id]
#             if marginal_id not in hs_keys_maps:
#                 hs_keys_maps[marginal_id] = []
#             hs_keys_maps[marginal_id].append(hs_sample_id)
#
#         for marginal_id in hs_keys_maps:
#             cols = self.kway_combinations[marginal_id]
#             hs_keys_ids = jnp.array(hs_keys_maps[marginal_id])
#             hs_keys = self.prefix_keys[hs_keys_ids].reshape(hs_keys_ids.shape[0], -1)
#             stat_fn_list.append(self.get_prefix_fn_helper(cols, hs_keys))
#             # stat_fn_list[st]
#
#         def stat_fn(X):
#             stats = jnp.concatenate([fn(X) for fn in stat_fn_list])
#             return stats
#         return stat_fn
#
#     def get_diff_stat_fn(self, stat_ids: list):
#         hs_keys_maps = {}
#         for i in stat_ids:
#             i = int(i)
#             marginal_id, hs_sample_id = self.prefix_map[i]
#             if marginal_id not in hs_keys_maps:
#                 hs_keys_maps[marginal_id] = []
#             hs_keys_maps[marginal_id].append(hs_sample_id)
#
#         stat_fn_list = []
#         for marginal_id in hs_keys_maps:
#             cols = self.kway_combinations[marginal_id]
#             hs_keys_ids = jnp.array(hs_keys_maps[marginal_id])
#             hs_keys = self.prefix_keys[hs_keys_ids].reshape(hs_keys_ids.shape[0], -1)
#             stat_fn_list.append(self.get_diff_prefix_fn_helper(cols, hs_keys))
#
#         def stat_fn(X, sigmoid):
#             stats = jnp.concatenate([fn(X, sigmoid) for fn in stat_fn_list])
#             return stats
#
#         return stat_fn
#
#     def get_sync_data_errors(self, X):
#         assert self.true_stats is not None, "Error: must call the fit function"
#         max_errors = []
#         for i in range(len(self.prefix_stats)):
#             fn_jit = self.prefix_fn_jit[i]
#             sync_stat = fn_jit(X).reshape((self.num_prefix_samples, -1))
#             true_stat = self.prefix_stats[i].reshape((self.num_prefix_samples, -1))
#             max_error = jnp.abs(true_stat - sync_stat).max(axis=1)
#             max_errors.append(max_error)
#         max_errors_jax = jnp.concatenate(max_errors)
#         return max_errors_jax
#
#     def _get_workload_fn(self, workload_ids=None):
#         """
#         Returns marginals function and sensitivity
#         :return:
#         """
#         dim = len(self.domain.attrs)
#         def answer_fn(x_row: chex.Array, query_single: chex.Array):
#             I = query_single[:self.k].astype(int)
#             U = query_single[self.k:2*self.k]
#             L = query_single[2*self.k:3*self.k]
#             t1 = (x_row[I] < U).astype(int)
#             t2 = (x_row[I] >= L).astype(int)
#             t3 = jnp.prod(jnp.array([t1, t2]), axis=0)
#             answers = jnp.prod(t3)
#             return answers
#
#         if workload_ids is None :
#             these_queries = self.queries
#         else:
#             these_queries = []
#             query_positions = []
#             for stat_id in workload_ids:
#                 a, b = self.workload_positions[stat_id]
#                 q_pos = jnp.arange(a, b)
#                 query_positions.append(q_pos)
#                 these_queries.append(self.queries[a:b, :])
#             these_queries = jnp.concatenate(these_queries, axis=0)
#
#         temp_rows_fn = jax.vmap(answer_fn, in_axes=(None, 0))
#         temp_stat_fn = jax.jit(jax.vmap(temp_rows_fn, in_axes=(0, None)))
#
#         def stat_fn(X):
#             return temp_stat_fn(X, these_queries).sum(0) / X.shape[0]
#
#         return stat_fn
#
#     # @jax.jit
#     # def get_halfspace
#     def get_prefix_fn_helper(self, cols: tuple, keys: chex.PRNGKey):
#         # Get categorical columns meta data
#         dim = len(self.domain.attrs)
#
#         num_hs_queries = keys.shape[0]
#
#         cat_cols = []
#         for col in cols:
#             if col in self.domain.get_categorical_cols():
#                 cat_cols.append(col)
#
#         cat_idx = self.domain.get_attribute_indices(cat_cols)
#         sizes = []
#         for col in cat_cols:
#             sizes.append(self.domain.size(col))
#         cat_idx = jnp.concatenate((cat_idx, jnp.array([dim]).astype(int))).astype(int)
#         sizes.append(1)
#         sizes_jnp = [jnp.arange(s + 1).astype(float) for i, s in zip(cat_idx, sizes)]
#
#         numeric_cols = self.domain.get_numeric_cols()
#         num_idx = self.domain.get_attribute_indices(numeric_cols)
#         numeric_dim = num_idx.shape[0]
#         kway_numeric = 2
#
#         # @jax.jit
#         def stat_fn(x_row, key):
#             # Compute statistic for a single row and halfspace
#             n=1
#             x_row = jnp.concatenate((x_row, jnp.ones(n).astype(int)))
#
#             # Cat
#             x_row_cat_proj = x_row[cat_idx].reshape((1, -1))
#             cat_answers = jnp.histogramdd(x_row_cat_proj, sizes_jnp)[0].flatten()
#
#             # Prefix
#             rng_h, rng_b = jax.random.split(key, 2)
#             thresholds = jax.random.uniform(rng_h, shape=(kway_numeric, )) # d x h
#             pos = jax.random.randint(rng_b, minval=0, maxval=numeric_dim, shape=(kway_numeric, )) # d x h
#             kway_idx = num_idx[pos]
#             below_threshold = (x_row[kway_idx] < thresholds).astype(int)  # n x d
#             prefix_answer = jnp.prod(below_threshold)
#
#             answers = cat_answers * prefix_answer
#             return answers
#
#         row_stat_fn_vmap = jax.vmap(stat_fn, in_axes=(None, 0))  # iterates over halfspaces
#         stat_fn_vmap = jax.vmap(row_stat_fn_vmap, in_axes=(0, None))  # Iterates over rows
#
#
#         def stat_fn_vmap2(X):
#             answers = stat_fn_vmap(X, keys)
#             return (answers.sum(0) / X.shape[0]).reshape(-1)
#
#         return stat_fn_vmap2
#
#     def get_diff_prefix_fn_helper(self, cols, keys: chex.PRNGKey):
#         cat_cols = []
#         for col in cols:
#             if col in self.domain.get_categorical_cols():
#                 cat_cols.append(col)
#         # For computing differentiable marginal queries
#         cat_queries = []
#         indices = [self.domain.get_attribute_onehot_indices(att) for att in cat_cols] + [jnp.array([-1])]
#         for tup in itertools.product(*indices):
#             cat_queries.append(tup)
#         cat_queries = jnp.array(cat_queries)
#
#         numeric_cols = self.domain.get_numeric_cols()
#         num_idx = jnp.array([self.domain.get_attribute_onehot_indices(att) for att in numeric_cols]).squeeze()
#         numeric_dim = num_idx.shape[0]
#         kway_numeric = 2
#
#
#         def stat_fn(x_row_onehot, sigmoid, key):
#             # X = jnp.column_stack((X, jnp.ones(n).astype(int)))
#             x_row_onehot = jnp.concatenate((x_row_onehot, jnp.ones(1).astype(int)))
#             cat_answers = jnp.prod(x_row_onehot[cat_queries], 1)
#
#             rng_h, rng_b = jax.random.split(key, 2)
#             thresholds = jax.random.uniform(rng_h, shape=(kway_numeric, )) # d x h
#             pos = jax.random.randint(rng_b, minval=0, maxval=numeric_dim, shape=(kway_numeric, )) # d x h
#             kway_idx = num_idx[pos]
#             below_threshold = jax.nn.sigmoid(-sigmoid * (x_row_onehot[kway_idx] - thresholds))  # n x d
#             prefix_answer = jnp.prod(below_threshold)
#
#             # diff_answers = jnp.multiply(cat_answers.reshape((n, -1, 1)), above_halfspace.reshape((n, 1, -1))).reshape((n, -1))
#             answers = cat_answers * prefix_answer
#
#             return answers
#         num_hs_queries = keys.shape[0]
#
#         stat_fn_vmap = jax.vmap(stat_fn, in_axes=(None, None, 0))
#         stat_fn_vmap = jax.vmap(stat_fn_vmap, in_axes=(0, None, None))  # Iterates over rows
#
#         def stat_fn_vmap2(X, sigmoid):
#             answers = stat_fn_vmap(X, sigmoid, keys)
#             return (answers.sum(0) / X.shape[0]).reshape(-1)
#
#         # stat_fn_vmap2 = lambda X, sigmoid: stat_fn_vmap(X, sigmoid, keys).reshape((num_hs_queries, -1))
#         # if num_hs_queries == 1:
#         #     stat_fn_vmap2 = lambda X, sigmoid: stat_fn_vmap(X, sigmoid, keys).reshape((-1))
#         # else:
#         #     stat_fn_vmap2 = lambda X, sigmoid: stat_fn_vmap(X, sigmoid, keys).reshape(-1)
#         return stat_fn_vmap2
#
#     def get_true_stats(self):
#         assert self.true_stats is not None, "Error: must call the fit function"
#         return jnp.concatenate([stat.flatten() for stat in self.prefix_stats])
#
#     def get_stats(self, data: Dataset, indices: list = None):
#         X = data.to_numpy()
#         stats = self.get_stats_jax(X)
#         return stats
#
#     def get_stats_jit(self, data: Dataset, indices: list = None):
#         X = data.to_numpy()
#         stats = self.get_stats_jax_jit(X)
#         return stats
#
#     def get_stats_jax(self, X: chex.Array):
#         total_num_rows = X.shape[0]
#         if total_num_rows <=2000:
#             stats = jnp.concatenate([fn(X).flatten() for fn in self.prefix_fn])
#         else:
#             prefix_stats = []
#             num_splits = total_num_rows // 2000
#             X_split = jnp.array_split(X, num_splits)
#             for fn in self.prefix_fn:
#                 stat_sum = None
#                 for i in range(num_splits):
#                     X_i = X_split[i]
#                     num_rows = X_i.shape[0]
#                     temp_stats = num_rows * fn(X_i)
#                     stat_sum = temp_stats if stat_sum is None else stat_sum + temp_stats
#                 hs_stats = stat_sum / total_num_rows
#                 prefix_stats.append(hs_stats)
#             stats = jnp.concatenate([stat.flatten() for stat in prefix_stats])
#         return stats
#
#     def get_stats_jax_jit(self, X: chex.Array):
#         stats = jnp.concatenate([fn(X).flatten() for fn in self.prefix_fn_jit])
#         return stats
#
#     def get_diff_halfspace_stats(self, data: Dataset, sigmoid, indices: list = None):
#         X = data.to_onehot()
#         stats = []
#         I = indices if indices is not None else list(range(self.get_num_workloads()))
#         for stat_fn_diff_jit in self.prefix_fn_diff_jit:
#             stats.append(stat_fn_diff_jit(X, sigmoid).flatten())
#         return jnp.concatenate(stats)
#
#
#
# ######################################################################
# ## TEST
# ######################################################################
# import pandas as pd
#
#
#
#
# def test_prefix():
#
#     print('debug')
#     cols = ['A', 'B', 'C', 'D', 'E', 'F']
#     domain = Domain(cols, [2, 5, 2, 1, 1, 1])
#
#     A = pd.DataFrame(np.array([
#         [0, 0, 1, 0.0, 0.1, 0.0],
#         [0, 2, 1, 0.2, 0.3, 0.0],
#         [0, 3, 1, 0.8, 0.9, 0.0],
#         [1, 1, 1, 0.1, 0.0, 0.0],
#         # [0, 1, 0.1, 0.0, 0.1],
#     ]), columns=cols)
#     data = Dataset(A, domain=domain)
#
#     rand_data = Dataset.synthetic(domain, N=10, seed=0)
#
#     cols = [('A',), ('B',), ('C', )]
#     # cols = []
#
#     num_random_halfspaces = 130
#     # cols = []
#     key = jax.random.PRNGKey(0)
#     stat_mod = Prefix(domain, kway_combinations=cols, rng=key, num_random_prefixes=num_random_halfspaces)
#     stat_mod.fit(data)
#
#     stat_fn = stat_mod.get_stat_fn(list(jnp.arange(stat_mod.get_num_workloads())))
#     diff_stat_fn = stat_mod.get_diff_stat_fn(list(jnp.arange(stat_mod.get_num_workloads())))
#     # print(stat_mod.get_true_stats())
#
#     stats = stat_fn(data.to_numpy())
#     diff_stats = diff_stat_fn(data.to_onehot(), 1000)
#
#     diff_error = jnp.abs(stats - diff_stats).max()
#     print(jnp.abs(stats - diff_stats).mean())
#     print('max diff error = ', diff_error)
#     assert diff_error < 0.001, f"Diff error is {diff_error:.6f}"
#
#     print(f'num queries={stat_mod.get_num_workloads()}')
#     for qid in range(stat_mod.get_num_workloads()):
#         stat_fn = stat_mod.get_stat_fn([qid])
#         stat = stat_mod.get_true_stat([qid])
#         diff = jnp.abs(stat - stat_fn(data.to_numpy()))
#         assert diff.max() < 0.01
#
#
# def test_fit_runtime():
#
#     print('debug')
#     cols_names = [f'f{i}' for i in range(20)]
#     cols_sizes = [3 for _ in range(10)] + [1 for _ in range(10)]
#     domain = Domain(cols_names, cols_sizes)
#     data = Dataset.synthetic(domain, N=10007, seed=0)
#     rand_data = Dataset.synthetic(domain, N=1000, seed=0)
#
#     t0 = time.time()
#     key = jax.random.PRNGKey(0)
#     stat_mod, _ = Prefix.get_kway_prefixes(domain, k=1, rng=key, random_prefixes=10003)
#     stat_mod.fit(data)
#     t1 = time.time()
#     print(f'fit time = {t1 - t0:.5f}')
#     stat_mod.get_true_stats()
#     temp = stat_mod.get_stats_jit(rand_data)
#     print(f'\tstat_size = {temp.shape}')
#     t2 = time.time()
#     print(f'get_stat_jit time = {t2 - t1:.5f}')
#     stat_mod.get_sync_data_errors(rand_data.to_numpy())
#     t3 = time.time()
#     print(f'get_sync_data_errors time = {t3 - t2:.5f}')
#
#
# if __name__ == "__main__":
#     # test_mixed()
#     # test_runtime()
#     # test_real_and_diff()
#     # test_discrete()
#     # test_row_answers()
#     test_prefix()
    # test_fit_runtime()