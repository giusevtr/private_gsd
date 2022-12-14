from utils import Dataset, Domain
import jax.numpy as jnp

class Statistic:

    def __init__(self, domain: Domain, name):
        self.domain = domain
        self.name = name

    def __str__(self):
        return self.name

    # def get_sub_statistics(self, index):
    #     true_stats = self.get_true_stats()
    #     return true_stats[index]

    def get_true_stats(self) -> jnp.array:
        pass

    def get_sensitivity(self):
        pass

    def get_stats_fn(self):
        pass

    def get_differentiable_stats_fn(self):
        pass
