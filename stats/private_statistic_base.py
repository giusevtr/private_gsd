from typing import Callable
import jax
from utils import Dataset, Domain
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class StatState:
    params: dict


class PrivateStatistic:
    privately_selected_statistics: list = None
    private_stats: jnp.ndarray = None
    confidence_bound: jnp.ndarray = None
    true_stats = jnp.ndarray

    def __init__(self, domain: Domain, name: str):
        # self.data = data
        self.domain = domain
        self.name = name


    def __str__(self):
        return self.name

    def fit(self, data: Dataset):
        pass

    def initialize(self) -> StatState:
        return StatState()

    def fit_private_stats(self, query_inx, priv_stats, confidence_bound):
        self.privately_selected_statistics = query_inx
        self.private_stats = priv_stats
        self.confidence_bound = confidence_bound

    def private_select_measure_statistic(self, key, rho, X_sync, error_type='max'):
        """
        Select a query with max error using the Exponential mechanism. Then privately evaluate the query
        using a private mechanism.
        """
        pass

    def true_sync_errors(self, X_sync, error_type: str, stat_fn: Callable):
        """
        Compute error against true statistics
        """
        pass

    def private_sync_errors(self, X_sync, error_type: str):
        """
        Compute error against the privatized statistics
        """
        pass

    def statistics(self, X: jnp.ndarray) -> jnp.ndarray:
        pass

    def differentiable_statistics(self, X: jnp.ndarray, sigmoid_param: float) -> jnp.ndarray:
        pass

    def get_num_queries(self) -> int:
        pass

    def get_true_stats(self) -> jnp.ndarray:
        pass

    def get_private_stats(self) -> jnp.ndarray:
        pass

    def get_dataset_size(self) -> int:
        return -1

    def get_sensitivity(self) -> float:
        pass

    # def get_sub_stat_module(self, indices: list):
    #     pass

    def get_stats_fn(self) -> Callable:
        pass

    def get_differentiable_stats_fn(self) -> Callable:
        pass

    def get_sub_stats_fn(self) -> Callable:
        pass

    def get_sub_differentiable_stats_fn(self) -> Callable:
        pass

    # def get_gaussian_mech_true_stats(self, key, rho: float):
    #     sigma_gaussian = float(jnp.sqrt(self.get_sensitivity() ** 2 / (2 * rho)))
    #     key, key_gaussian = jax.random.split(key, 2)
    #     true_stats = self.get_true_stats()
    #     true_stats_noise = self.get_true_stats() + jax.random.normal(key_gaussian, shape=true_stats.shape) * sigma_gaussian
    #     return true_stats_noise
    #
    # def setup_constrain(self, stat_fn: Callable[[jnp.ndarray], jnp.ndarray],
    #                      target_stats: jnp.ndarray, constrain_width: float):
    #     self.constrain_stat_fn = stat_fn
    #     self.constrain_stats = target_stats
    #     self.constrain_width = constrain_width
    #
    # def constrain_fn(self, X: jnp.ndarray):
    #     eval = jnp.abs(self.constrain_stats - self.constrain_stat_fn(X))
    #     cont = jnp.where(eval > self.constrain_width, eval + 1, 0)
    #     return cont
