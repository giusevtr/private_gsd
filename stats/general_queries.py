import itertools
import jax
import jax.numpy as jnp
import chex
from utils import Dataset, Domain
from stats import AdaptiveStatisticState
from tqdm import tqdm
import numpy as np


class GeneralQuery(AdaptiveStatisticState):

    def __init__(self, domain, query_list: list):
        self.domain = domain
        self.query_list = query_list
        self.workload_positions = []
        self.workload_sensitivity = []

    def __str__(self):
        return f'General Queries'

    def get_num_workloads(self):
        return len(self.query_list)

    def _get_workload_positions(self, workload_id: int = None) -> tuple:
        return (workload_id, workload_id+1)

    def is_workload_numeric(self, cols):
        for c in cols:
            if c in self.domain.get_numeric_cols():
                return True
        return False

    def _get_workload_sensitivity(self, workload_id: int = None, N: int = None) -> float:
        return 1 / N

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
        if workload_ids is None:
            query_ids = list(np.arange(len(self.query_list)))
        else:
            query_ids = workload_ids

        temp_queries = [self.query_list[i] for i in query_ids]

        def stat_fn(X):
            ans = [query(X) for query in temp_queries]
            return jnp.concatenate(ans)

        return stat_fn

    # def _get_stat_fn(self, query_ids):
    #     def answer_fn(x_row: chex.Array, query_single: chex.Array):
    #         query_single = int(query_single)
    #         answers = self.query_list[query_single](x_row)
    #         return answers
    #
    #     these_queries = jnp.array(query_ids)
    #     temp_stat_fn = jax.vmap(answer_fn, in_axes=(None, 0))
    #
    #     def scan_fun(carry, x):
    #         return carry + temp_stat_fn(x, these_queries), None
    #     def stat_fn(X):
    #         out = jax.eval_shape(temp_stat_fn, X[0], these_queries)
    #         stats = jax.lax.scan(scan_fun, jnp.zeros(out.shape, out.dtype), X)[0]
    #         return stats / X.shape[0]
    #     return stat_fn

