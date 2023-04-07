import itertools
import jax
import jax.numpy as jnp
import chex
from utils import Dataset, Domain
from stats import AdaptiveStatisticState
from tqdm import tqdm
import numpy as np


class HalfspacesBT(AdaptiveStatisticState):

    def __init__(self, domain, key: chex.PRNGKey, random_proj: int, bins=(32,)):
        self.domain = domain
        self.key = key
        self.random_proj = random_proj
        self.bins = list(bins)
        self.workload_positions = []
        self.workload_sensitivity = []

        self.cat_pos = domain.get_attribute_indices(domain.get_categorical_cols())
        self.cat_sz = [domain.size(cat) for cat in domain.get_categorical_cols()]
        self.num_pos = domain.get_attribute_indices(domain.get_numeric_cols())

        self.set_up_stats()

    def __str__(self):
        return f'Halfspaces-BT'

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
        key = self.key

        queries = []
        self.workload_positions = []
        self.proj_keys = jax.random.split(key)
        for proj_id in tqdm(range(self.random_proj), desc='Setting up Halfspaces-BT.'):
            # key, key_sub = jax.random.split(key)

            # indices = self.domain.get_attribute_indices(marginal)
            # bins = self.bins if self.is_workload_numeric(marginal) else [-1]
            start_pos = len(queries)
            for bin in self.bins:
                intervals = []
                upper = np.linspace(-1, 1, num=bin+1)[1:]
                lower = np.linspace(-1, 1, num=bin+1)[:-1]
                lower[0] = -100.01
                upper[-1] = 100.01
                # upper = upper.at[-1].set(1.01)
                interval = list(np.vstack((upper, lower)).T)
                intervals.append(interval)
                for v in itertools.product(*intervals):
                    v_arr = np.array(v)
                    up = v_arr.flatten()[::2]
                    lo = v_arr.flatten()[1::2]
                    q = np.concatenate((jnp.array([proj_id]), up, lo))
                    queries.append(q)  # (key), ((a1, a2), (b1, b2))
            end_pos = len(queries)
            self.workload_positions.append((start_pos, end_pos))
            self.workload_sensitivity.append(jnp.sqrt(2 * len(self.bins)))

        self.queries = jnp.array(queries)

    def _get_workload_sensitivity(self, workload_id: int = None, N: int = None) -> float:
        return self.workload_sensitivity[workload_id] / N

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
        # query_ids = []
        if workload_ids is None:
        #     these_queries = self.queries
            query_ids = jnp.arange(self.queries.shape[0])
        else:
            query_positions = []
            for stat_id in workload_ids:
                a, b = self.workload_positions[stat_id]
                q_pos = jnp.arange(a, b)
                query_positions.append(q_pos)
            query_ids = jnp.concatenate(query_positions)

        return self._get_stat_fn(query_ids)

    def random_linear_proj(self, key: chex.PRNGKey, x: chex.Array):
        sum = 0
        norm = jnp.sqrt(x.shape[0])
        for pos, sz in zip(self.cat_pos, self.cat_sz):
            key, key_sub = jax.random.split(key)
            h = jax.random.normal(key_sub, shape=(sz, )) / norm

            x_val = x[pos].astype(int)
            sum += h[x_val]

        key, key_sub = jax.random.split(key)
        h = jax.random.normal(key_sub, shape=(self.num_pos.shape[0], )) / norm
        num_proj = x[self.num_pos]
        sum += jnp.dot(h, num_proj)
        return sum


    def answer_fn(self, x_row: chex.Array, query_single: chex.Array):
        key_id = query_single[0].astype(jnp.uint32)
        key = self.proj_keys[key_id]
        x_proj = self.random_linear_proj(key, x_row)
        hi = query_single[1]
        lo = query_single[2]
        answers1 = lo < x_proj
        answers2 = x_proj < hi
        return (answers1 * answers2).astype(float)

    def _get_stat_fn(self, query_ids):

        these_queries = self.queries[query_ids]
        temp_stat_fn = jax.vmap(self.answer_fn, in_axes=(None, 0))

        def scan_fun(carry, x):
            return carry + temp_stat_fn(x, these_queries), None
        def stat_fn(X):
            out = jax.eval_shape(temp_stat_fn, X[0], these_queries)
            stats = jax.lax.scan(scan_fun, jnp.zeros(out.shape, out.dtype), X)[0]
            return stats / X.shape[0]
        return stat_fn

    # @staticmethod
    # def get_kway_categorical(domain: Domain, k):
    #     kway_combinations = [list(idx) for idx in itertools.combinations(domain.get_categorical_cols(), k)]
    #     return Marginals(domain, kway_combinations, k, bins=[2])
    #
    # @staticmethod
    # def get_all_kway_combinations(domain, k, bins=(32,)):
    #     kway_combinations = [list(idx) for idx in itertools.combinations(domain.attrs, k)]
    #     return Marginals(domain, kway_combinations, k, bins=bins)
    #
    # @staticmethod
    # def get_all_kway_mixed_combinations_v1(domain, k, bins=(32,)):
    #     num_numeric_feats = len(domain.get_numeric_cols())
    #     k_real = num_numeric_feats
    #     kway_combinations = []
    #     for cols in itertools.combinations(domain.attrs, k):
    #         count_disc = 0
    #         count_real = 0
    #         for c in list(cols):
    #             if c in domain.get_numeric_cols():
    #                 count_real += 1
    #             else:
    #                 count_disc += 1
    #         if count_disc > 0 and count_real > 0:
    #             kway_combinations.append(list(cols))
    #
    #     return Marginals(domain, kway_combinations, k, bins=bins)




if __name__ == "__main__":

    data = Dataset.synthetic(Domain(['A', 'B', 'C'], [1, 2, 3]), N=1000, seed=0)
    data_np = data.to_numpy()
    module = HalfspacesBT(data.domain, key=jax.random.PRNGKey(0), random_proj=1, bins=(2, 4, 8, 16))

    module.answer_fn(data_np[0], module.queries[0])

    temp_fn = module._get_stat_fn(jnp.array([0]))
    stat = temp_fn(data_np)
    print(stat)



    stat_fn = module._get_dataset_statistics_fn()
    stats = stat_fn(data)
    print(stats)
    print(stats.sum())