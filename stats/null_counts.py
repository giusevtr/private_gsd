import itertools
import jax
import jax.numpy as jnp
import chex
from utils import Dataset, Domain
from stats import AdaptiveStatisticState
from tqdm import tqdm
import numpy as np


class NullCounts(AdaptiveStatisticState):

    def __init__(self, domain, null_cols = None):
        self.domain = domain
        self.workload_positions = []
        self.workload_sensitivity = []
        self.set_up_stats()

        self.null_cols = self.domain.attrs if null_cols is None else null_cols

    def __str__(self):
        return f'NullCounts'

    def get_num_workloads(self):
        dim = len(self.null_cols)
        return dim

    def _get_workload_positions(self, workload_id: int = None) -> tuple:
        return (workload_id, workload_id+1)

    def set_up_stats(self):
        pass

    def _get_workload_sensitivity(self, workload_id: int = None, N: int = None) -> float:
        # dim = len(self.domain.attrs)
        return jnp.sqrt(2) / N

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
        """
        Returns marginals function and sensitivity
        :return:
        """

        def answer_fn(x_row: chex.Array, col_index: chex.Array):
            return jnp.isnan(x_row[col_index]).astype(int)

        # if workload_ids is None:
        #     these_queries = self.queries
        # else:
        #     these_queries = []
        #     query_positions = []
        #     for stat_id in workload_ids:
        #         a, b = self.workload_positions[stat_id]
        #         q_pos = jnp.arange(a, b)
        #         query_positions.append(q_pos)
        #         these_queries.append(self.queries[a:b, :])
        #     these_queries = jnp.concatenate(these_queries, axis=0)
        temp_stat_fn = jax.vmap(answer_fn, in_axes=(None, 0))
        if workload_ids is None:
            dim = len(self.domain.attrs)
            # attrs_indices = jnp.arange(dim)
            null_attrs = self.null_cols
        else:
            # attrs_indices = jnp.array(workload_ids)
            null_attrs = [self.null_cols[w_id] for w_id in workload_ids]

        attrs_indices = self.domain.get_attribute_indices(null_attrs)


        def scan_fun(carry, x):
            return carry + temp_stat_fn(x, attrs_indices), None
        def stat_fn(X):
            out = jax.eval_shape(temp_stat_fn, X[0], attrs_indices)
            stats = jax.lax.scan(scan_fun, jnp.zeros(out.shape, out.dtype), X)[0]
            return stats / X.shape[0]

        return stat_fn


