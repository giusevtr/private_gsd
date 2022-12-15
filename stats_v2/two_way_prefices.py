import itertools
import jax.numpy as jnp
import jax
from utils import Dataset, Domain
from stats_v2 import Statistic

NAME = 'Rand 2-way Prefix'

def get_stats_fn(num_cols, num_rand_queries, seed=0):
    assert num_cols >= 2
    key = jax.random.PRNGKey(seed)
    key1, key2 = jax.random.split(key)
    thresholds = jax.random.uniform(key1, shape=(num_rand_queries, 2))
    cols = jax.random.choice(key2, jnp.arange(num_cols), shape=(num_rand_queries, 2), replace=True, axis=0)
    cols1 = cols[:, 0]
    cols2 = cols[:, 1]

    def stat_fn(X):
        answers_1 = 1 * (X[:, cols1] < thresholds[:, 0])
        answers_2 = 1 * (X[:, cols2] < thresholds[:, 1])
        answers = jnp.prod(jnp.stack([answers_1, answers_2]), axis=0)
        stats = answers.mean(axis=0)
        return stats

    return stat_fn

from dataclasses import dataclass

class TwoWayPrefix(Statistic):
    # columns: jnp.ndarray
    # thresholds: jnp.ndarray
    true_stats: list

    def __init__(self, domain, columns, thresholds, name='Two-way prefix'):
        super().__init__(domain, name)

        self.columns = columns
        self.thresholds = thresholds

    @staticmethod
    def get_stat_module(domain, num_rand_queries, seed=0):
        # Get real-valued columns
        real_valued_cols = domain.get_numeric_cols()
        real_valued_index = domain.get_attribute_indices(real_valued_cols)

        key = jax.random.PRNGKey(seed)
        key1, key2 = jax.random.split(key)
        cols = jax.random.choice(key1, real_valued_index, shape=(num_rand_queries, 2), replace=True, axis=0)
        thresholds = jax.random.uniform(key2, shape=(num_rand_queries, 2))
        # self.cols1 = self.cols[:, 0]
        # self.cols2 = self.cols[:, 1]
        return TwoWayPrefix(domain, cols, thresholds)

    def get_num_queries(self):
        return self.columns.shape[0]

    def get_sub_stat_module(self, indices):
        return TwoWayPrefix(self.domain, self.columns[indices, :], self.thresholds[indices, :])

    def get_sensitivity(self):
        return 1


    def get_stats_fn(self):
        cols1_sub = self.columns[:, 0]
        cols2_sub = self.columns[:, 1]
        thres1_sub = self.thresholds[:, 0]
        thres2_sub = self.thresholds[:, 1]

        @jax.jit
        def stat_fn(X):
            answers_1 = 1 * (X[:, cols1_sub] < thres1_sub)
            answers_2 = 1 * (X[:, cols2_sub] < thres2_sub)
            answers = jnp.prod(jnp.stack([answers_1, answers_2]), axis=0)
            stats = answers.mean(axis=0)
            return stats
        return stat_fn


    def get_differentiable_stats_fn(self):
        cols1_sub = self.columns[:, 0]
        cols2_sub = self.columns[:, 1]
        thres1_sub = self.thresholds[:, 0]
        thres2_sub = self.thresholds[:, 1]

        @jax.jit
        def diff_stat_fn(X, sigmoid):
            answers_1 = jax.nn.sigmoid(sigmoid * (- X[:, cols1_sub] + thres1_sub))
            answers_2 = jax.nn.sigmoid(sigmoid * (-X[:, cols2_sub] + thres2_sub))
            answers = jnp.prod(jnp.stack([answers_1, answers_2]), axis=0)
            stats = answers.mean(axis=0)
            return stats

        return diff_stat_fn
