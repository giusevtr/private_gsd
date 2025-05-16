import jax
import jax.numpy as jnp
import chex
from math import sqrt
from genetic_sd.adaptive_statistics import AdaptiveStatisticState
from genetic_sd.utils import Dataset, Domain


class SparseCounts(AdaptiveStatisticState):
    """
    This module calculates the sparsity rate of a sparse-column.
    """

    def __init__(self, domain: Domain):
        self.domain = domain
        self.workload_positions = []
        self.workload_sensitivity = []
        self.set_up_stats()

        # self.null_cols = self.domain.attrs if null_cols is None else null_cols
        self.sparse_cols = []
        # self.sparse_value = []

        for col in domain.attrs:
            if domain.is_sparse(col):
                self.sparse_cols.append(col)

    def __str__(self):
        return f'Sparse counts'

    def get_num_workloads(self):
        dim = len(self.sparse_cols)
        return dim

    def _get_workload_positions(self, workload_id: int = None) -> tuple:
        return (workload_id, workload_id+1)

    def set_up_stats(self):
        pass

    def _get_workload_sensitivity(self, workload_id: int = None, N: int = None) -> float:
        # dim = len(self.domain.attrs)
        return sqrt(2) / N

    def _get_dataset_statistics_fn(self, workload_ids=None, jitted: bool = False):
        if jitted:
            workload_fn = jax.jit(self._get_workload_fn(workload_ids))
        else:
            workload_fn = self._get_workload_fn(workload_ids)

        def data_fn(data: Dataset):
            X = data.to_numpy()
            return workload_fn(X)
        return data_fn

    def _get_workload_fn(self, workload_ids=None):
        return self._get_stat_fn(workload_ids)

    def _get_stat_fn(self, query_ids):

        # Define the sparse function
        def answer_fn(x_row: chex.Array, col_index: chex.Array):
            sparse_value = 0
            is_sparse = jnp.abs(x_row[col_index] - sparse_value) < 1e-9
            return is_sparse.astype(int)

        temp_stat_fn = jax.vmap(answer_fn, in_axes=(None, 0))
        if query_ids is None:
            null_attrs = self.sparse_cols
        else:
            # attrs_indices = jnp.array(workload_ids)
            null_attrs = [self.sparse_cols[w_id] for w_id in query_ids]
        attrs_indices = self.domain.get_attribute_indices(null_attrs)

        def scan_fun(carry, x):
            return carry + temp_stat_fn(x, attrs_indices), None

        def stat_fn(X):
            out = jax.eval_shape(temp_stat_fn, X[0], attrs_indices)
            stats = jax.lax.scan(scan_fun, jnp.zeros(out.shape, out.dtype), X)[0]
            return stats / X.shape[0]

        return stat_fn
