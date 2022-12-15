from typing import Callable

from utils import Dataset, Domain
import jax.numpy as jnp

class Statistic:

    def __init__(self, domain: Domain, name):
        # self.data = data
        self.domain = domain
        self.name = name
        self.true_stats = None
        self.stat_fn: Callable
        self.diff_fn = None

    def __str__(self):
        return self.name

    def fit(self, data: Dataset):
        pass

    def get_num_queries(self):
        pass

    def get_true_stats(self) -> jnp.array:
        pass

    def get_sub_true_stats(self, index: list):
        pass

    def get_sensitivity(self):
        pass

    def get_stats_fn(self) -> Callable:
        pass

    def get_differentiable_stats_fn(self) -> Callable:
        pass

    def get_sub_stat_module(self, indices: list):
        pass

    def get_sync_data_errors(self, X):
        pass

    def get_dataset_size(self):
        return -1
