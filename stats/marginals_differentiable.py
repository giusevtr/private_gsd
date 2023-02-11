import itertools
import jax
import jax.numpy as jnp
from utils import Dataset, Domain
from stats import AdaptiveStatisticState
from tqdm import tqdm


class MarginalsDiff(AdaptiveStatisticState):

    def __init__(self, domain, kway_combinations, k, bins=(32,)):
        self.domain = domain
        self.kway_combinations = kway_combinations
        self.k = k
        self.bins = list(bins)
        self.workload_positions = []
        self.workload_sensitivity = []
        self.set_up_stats()

    def __str__(self):
        return f'Marginals(Differentiable)'

    def get_num_workloads(self):
        return len(self.workload_positions)

    def _get_workload_positions(self, workload_id: int = None) -> tuple:
        return self.workload_positions[workload_id]

    def is_workload_numeric(self, cols):
        for c in cols:
            if c in self.domain.get_numeric_cols():
                return True
        return False

    def set_up_stats(self):

        queries = []
        self.workload_positions = []
        for marginal in tqdm(self.kway_combinations, desc='Setting up Differentiable Marginals.'):
            assert len(marginal) == self.k
            indices = self.domain.get_attribute_indices(marginal)
            indices_onehot = [self.domain.get_attribute_onehot_indices(att) for att in marginal]
            start_pos = len(queries)
            for tup in itertools.product(*indices_onehot):
                queries.append(tup)
            end_pos = len(queries)
            self.workload_positions.append((start_pos, end_pos))
            self.workload_sensitivity.append(jnp.sqrt(2))

        self.queries = jnp.array(queries)

    def _get_workload_sensitivity(self, workload_id: int = None, N: int = None) -> float:
        return self.workload_sensitivity[workload_id] / N

    def _get_dataset_statistics_fn(self, workload_ids=None):
        workload_fn = self._get_workload_fn(workload_ids)
        def data_fn(data: Dataset):
            return workload_fn(data.to_onehot() )
        return data_fn

    def _get_workload_fn(self, workload_ids=None):

        if workload_ids is None:
            these_queries = self.queries
        else:
            these_queries = []
            query_positions = []
            for stat_id in workload_ids:
                a, b = self.workload_positions[stat_id]
                q_pos = jnp.arange(a, b)
                query_positions.append(q_pos)
                these_queries.append(self.queries[a:b, :])
            these_queries = jnp.concatenate(these_queries, axis=0)


        def row_diff_stat_fn(x, diff_query):
            return jnp.prod(x[diff_query])

        temp_stat_fn = jax.vmap(row_diff_stat_fn, in_axes=(None, 0))

        def scan_fun(carry, x):
            return carry + temp_stat_fn(x, these_queries), None

        def stat_fn(X):
            out = jax.eval_shape(temp_stat_fn, X[0], these_queries)
            stats = jax.lax.scan(scan_fun, jnp.zeros(out.shape, out.dtype), X)[0]
            return stats / X.shape[0]

        return stat_fn



    @staticmethod
    def get_all_kway_combinations(domain, k, bins=(32,)):
        kway_combinations = [list(idx) for idx in itertools.combinations(domain.attrs, k)]
        return MarginalsDiff(domain, kway_combinations, k, bins=bins)

    @staticmethod
    def get_all_kway_mixed_combinations_v1(domain, k, bins=(32,)):
        num_numeric_feats = len(domain.get_numeric_cols())
        k_real = num_numeric_feats
        kway_combinations = []
        for cols in itertools.combinations(domain.attrs, k):
            count_disc = 0
            count_real = 0
            for c in list(cols):
                if c in domain.get_numeric_cols():
                    count_real += 1
                else:
                    count_disc += 1
            if count_disc > 0 and count_real > 0:
                kway_combinations.append(list(cols))

        return MarginalsDiff(domain, kway_combinations, k, bins=bins)


######################################################################
## TEST
######################################################################
