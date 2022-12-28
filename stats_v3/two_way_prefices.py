import itertools
import jax.numpy as jnp
import jax
from utils import Dataset, Domain
from stats_v2 import Statistic
from typing import Callable
NAME = 'Rand 2-way Prefix'


class TwoWayPrefix(Statistic):
    # columns: jnp.ndarray
    # thresholds: jnp.ndarray
    true_stats: jnp.ndarray
    N: int = None  # Dataset size
    stat_fn: Callable[[jnp.array], jnp.ndarray] = None
    stat_fn_jit:  Callable[[jnp.array], jnp.ndarray] = None

    def __init__(self, domain, columns, thresholds, name='Two-way prefix'):
        super().__init__(domain, name)

        self.columns = columns
        self.thresholds = thresholds

    @staticmethod
    def get_stat_module(domain: Domain, num_rand_queries: int, seed=0):
        # Get real-valued columns
        real_valued_cols = domain.get_numeric_cols()
        real_valued_index = domain.get_attribute_indices(real_valued_cols)

        key = jax.random.PRNGKey(seed)
        key1, key2 = jax.random.split(key)
        cols = jax.random.choice(key1, real_valued_index, shape=(num_rand_queries, 2), replace=True, axis=0)
        thresholds = jax.random.uniform(key2, shape=(num_rand_queries, 2))
        return TwoWayPrefix(domain, cols, thresholds)

    def fit(self, data: Dataset):
        X = data.to_numpy()
        self.N = X.shape[0]
        self.stat_fn = self.get_stats_fn()
        self.stat_fn_jit = jax.jit(self.get_stats_fn())
        self.true_stats = self.stat_fn(X)

    def get_num_queries(self):
        return self.columns.shape[0]

    # def get_sub_stat_module(self, indices: list):
    #     indices = jnp.array(indices)
    #     sub_module = TwoWayPrefix(self.domain, self.columns[indices, :], self.thresholds[indices, :])
    #     sub_module.stat_fn = sub_module.get_stats_fn()
    #     sub_module.stat_fn_jit = jax.jit(sub_module.get_stats_fn())
    #     sub_module.N = self.N
    #     sub_module.true_stats = self.true_stats[indices]
    #     return sub_module

    def get_dataset_size(self):
        return self.N

    def get_sensitivity(self):
        return 1 / self.N

    def get_sub_true_stats(self, index: list) -> jnp.ndarray:
        return self.true_stats[jnp.array(index)]

    def get_true_stats(self) -> jnp.ndarray:
        return self.true_stats

    def get_sync_data_errors(self, X):
        sync_stats = self.stat_fn(X)
        return jnp.abs(self.get_true_stats() - sync_stats)

    def get_stats_fn(self):
        return self.get_sub_stats_fn_help(jnp.arange(self.columns.shape[0]))

    def get_differentiable_stats_fn(self):
        return self.get_differentiable_stats_fn_help(jnp.arange(self.columns.shape[0]))

    def get_sub_stats_fn(self) -> Callable:
        idx = jnp.array(self.privately_selected_statistics)
        return self.get_sub_stats_fn_help(idx)

    def get_sub_differentiable_stats_fn(self) -> Callable:
        idx = jnp.array(self.privately_selected_statistics)
        return self.get_differentiable_stats_fn_help(idx)

    def get_sub_stats_fn_help(self, idx) -> Callable:
        # self.columns[indices, :], self.thresholds[indices, :]
        # pass

        cols1_sub = self.columns[idx, 0]
        cols2_sub = self.columns[idx, 1]
        thres1_sub = self.thresholds[idx, 0]
        thres2_sub = self.thresholds[idx, 1]

        # @jax.jit
        def stat_fn(X):
            answers_1 = 1 * (X[:, cols1_sub] < thres1_sub)
            answers_2 = 1 * (X[:, cols2_sub] < thres2_sub)
            answers = jnp.prod(jnp.stack([answers_1, answers_2]), axis=0)
            stats = answers.mean(axis=0)
            return stats
        return stat_fn


    def get_differentiable_stats_fn_help(self, idx):
        cols1_sub = self.columns[idx, 0]
        cols2_sub = self.columns[idx, 1]
        thres1_sub = self.thresholds[idx, 0]
        thres2_sub = self.thresholds[idx, 1]

        @jax.jit
        def diff_stat_fn(X, sigmoid):
            answers_1 = jax.nn.sigmoid(sigmoid * (- X[:, cols1_sub] + thres1_sub))
            answers_2 = jax.nn.sigmoid(sigmoid * (-X[:, cols2_sub] + thres2_sub))
            answers = jnp.prod(jnp.stack([answers_1, answers_2]), axis=0)
            stats = answers.mean(axis=0)
            return stats

        return diff_stat_fn



######################################################################
# Test constrain
# def test_constrain():
#
#     from toy_datasets.circles import get_circles_dataset
#
#     data1 = get_circles_dataset(DATA_SIZE=10, seed=0)
#     data2 = get_circles_dataset(DATA_SIZE=10, seed=1)
#
#     stat_module = TwoWayPrefix.get_stat_module(data2.domain, 100)
#
#     stat_fn = stat_module.get_stats_fn()
#     stats = stat_fn(data1.to_numpy())
#     stat_module.setup_constrain(stat_fn, stats, 0.01)
#
#     const = stat_module.constrain_fn(data2.to_numpy())
#     print(const)
#
# if __name__ == "__main__":
#     test_constrain()

