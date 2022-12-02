import jax
from utils import Domain
from stats import Statistic


class Generator:
    def __init__(self, domain: Domain, stat_module: Statistic, data_size, seed):
        self.key = jax.random.PRNGKey(seed)
        self.domain = domain
        self.stat_module = stat_module
        self.data_size = data_size
        self.data_dim = domain.get_dimension()

    @staticmethod
    def get_generator(self):
        pass

    def fit(self, true_stats, init_X=None):
        pass



