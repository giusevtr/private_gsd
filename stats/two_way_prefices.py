import itertools
import jax.numpy as jnp
import jax
from utils import Dataset, Domain
from stats import Statistic

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
        answers_1 = 1 *(X[:, cols1] < thresholds[:, 0])
        answers_2 = 1 *(X[:, cols2] < thresholds[:, 1])
        answers = jnp.prod(jnp.stack([answers_1, answers_2]), axis=0)
        stats = answers.mean(axis=0)
        return stats

    return stat_fn

class TwoWayPrefix(Statistic):
    def __init__(self, domain, num_rand_queries, seed=0):
        super().__init__(domain, name=f'Two Way Prefix({num_rand_queries})')
        self.num_cols = len(domain.attrs)
        self.num_rand_queries = num_rand_queries
        self.seed = seed

        # Get real-valued columns
        real_valued_cols = domain.get_numeric_cols()
        real_valued_index = domain.get_attribute_indices(real_valued_cols)

        key = jax.random.PRNGKey(self.seed)
        key1, key2 = jax.random.split(key)
        self.thresholds = jax.random.uniform(key1, shape=(self.num_rand_queries, 2))
        self.cols = jax.random.choice(key2, real_valued_index, shape=(num_rand_queries, 2), replace=True, axis=0)
        self.cols1 = self.cols[:, 0]
        self.cols2 = self.cols[:, 1]

    def get_sensitivity(self):
        return jnp.sqrt(self.num_rand_queries)

    def get_stats_fn(self):

        def stat_fn(X):
            answers_1 = 1 * (X[:, self.cols1] < self.thresholds[:, 0])
            answers_2 = 1 * (X[:, self.cols2] < self.thresholds[:, 1])
            answers = jnp.prod(jnp.stack([answers_1, answers_2]), axis=0)
            stats = answers.mean(axis=0)
            return stats
        return  stat_fn

    def get_differentiable_stats_fn(self):
        def diff_stat_fn(X, sigmoid):
            answers_1 = jax.nn.sigmoid(sigmoid * (- X[:, self.cols1] + self.thresholds[:, 0]))
            answers_2 = jax.nn.sigmoid(sigmoid * (-X[:, self.cols2] + self.thresholds[:, 1]))
            answers = jnp.prod(jnp.stack([answers_1, answers_2]), axis=0)
            stats = answers.mean(axis=0)
            return stats

        return diff_stat_fn
