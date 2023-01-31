import itertools
import jax
import jax.numpy as jnp
from utils import Dataset, Domain
from utils.utils_data import Domain
import chex
from stats import AdaptiveStatisticState
from tqdm import tqdm

class Marginals(AdaptiveStatisticState):


    def __init__(self, domain, kway_combinations, k, bins=(32,)):
        self.domain = domain
        self.kway_combinations = kway_combinations
        self.k = k
        self.bins = list(bins)
        self.workload_positions = []
        self.workload_sensitivity = []

        self.set_up_stats()

    def __str__(self):
        return f'Marginals'

    def get_num_workloads(self):
        return len(self.workload_positions)

    def is_workload_numeric(self, cols):
        for c in cols:
            if c in self.domain.get_numeric_cols():
                return True
        return False

    def set_up_stats(self):

        queries = []
        self.workload_positions = []
        for marginal in tqdm(self.kway_combinations, desc='Setting up Marginals.'):
            assert len(marginal) == self.k
            indices = self.domain.get_attribute_indices(marginal)
            bins = self.bins if self.is_workload_numeric(marginal) else [-1]
            start_pos = len(queries)
            for bin in bins:
                intervals = []
                for att in marginal:
                    size = self.domain.size(att)
                    if size > 1:
                        upper = jnp.linspace(0, size, num=size+1)[1:]
                        lower = jnp.linspace(0, size, num=size+1)[:-1]
                        # lower = lower.at[0].set(-0.01)
                        interval = list(jnp.vstack((upper, lower)).T - 0.1)
                        intervals.append(interval)
                    else:
                        upper = jnp.linspace(0, 1, num=bin+1)[1:]
                        lower = jnp.linspace(0, 1, num=bin+1)[:-1]
                        upper = upper.at[-1].set(1.01)
                        interval = list(jnp.vstack((upper, lower)).T)
                        intervals.append(interval)
                for v in itertools.product(*intervals):
                    v_arr = jnp.array(v)
                    upper = v_arr.flatten()[::2]
                    lower = v_arr.flatten()[1::2]
                    q = jnp.concatenate((indices, upper, lower))
                    queries.append(q)  # (i1, i2), ((a1, a2), (b1, b2))
            end_pos = len(queries)
            self.workload_positions.append((start_pos, end_pos))
            self.workload_sensitivity.append(jnp.sqrt(2))

        self.queries = jnp.array(queries)

    def _get_workload_sensitivity(self, workload_id: int = None, N: int = None) -> float:
        return self.workload_sensitivity[workload_id] / N

    def _get_workload_fn(self, workload_ids=None):
        """
        Returns marginals function and sensitivity
        :return:
        """
        dim = len(self.domain.attrs)
        def answer_fn(x_row: chex.Array, query_single: chex.Array):
            I = query_single[:self.k].astype(int)
            U = query_single[self.k:2*self.k]
            L = query_single[2*self.k:3*self.k]
            t1 = (x_row[I] < U).astype(int)
            t2 = (x_row[I] >= L).astype(int)
            t3 = jnp.prod(jnp.array([t1, t2]), axis=0)
            answers = jnp.prod(t3)
            return answers

        if workload_ids is None :
            these_queries = self.queries
        else:
            these_queries = []
            for stat_id in workload_ids:
                a, b = self.workload_positions[stat_id]
                these_queries.append(self.queries[a:b, :])
        # queries = jnp.concatenate([self.queries[a:b, :] for (a, b) in self.workload_positions])
            these_queries = jnp.concatenate(these_queries, axis=0)
        temp_rows_fn = jax.vmap(answer_fn, in_axes=(None, 0))
        temp_stat_fn = jax.jit(jax.vmap(temp_rows_fn, in_axes=(0, None)))
        # stat_fn = lambda X: temp_stat_fn(X, queries)

        def stat_fn(X):
            return temp_stat_fn(X, these_queries).sum(0) / X.shape[0]

        return stat_fn




    @staticmethod
    def get_all_kway_combinations(domain, k, bins=(32,)):
        kway_combinations = [list(idx) for idx in itertools.combinations(domain.attrs, k)]
        return Marginals(domain, kway_combinations, k, bins=bins), kway_combinations

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

        return Marginals(domain, kway_combinations, k, bins=bins), kway_combinations




######################################################################
## TEST
######################################################################


def test_discrete():
    import pandas as pd
    cols = ['A', 'B', 'C']
    dom = Domain(cols, [2, 2, 2])

    raw_data_array = pd.DataFrame([
                        [0, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0]], columns=cols)
    # data = Dataset.synthetic_rng(dom, data_size, rng)
    data = Dataset(raw_data_array, dom)
    numeric_features = data.domain.get_numeric_cols()

    stat_mod, _ = Marginals.get_all_kway_combinations(data.domain, 2)
    stat_mod.fit(data)

    stat_fn = stat_mod.get_categorical_marginal_stats_fn_helper()
    stats1 = stat_fn(data.to_numpy())
    print(stats1)
    print('test_discrete() passed!')

def test_mixed():
    import pandas as pd
    cols = ['A', 'B']
    dom = Domain(cols, [3, 1])

    raw_data_array = pd.DataFrame([
                        [0, 0.0],
                        [0, 0.5],
                        [0, 1.0],
                        [1, 0.0],
                        [1, 0.5],
                        [1, 1.0],
                        # [1, 0.2]
    ],
        columns=cols)
    # data = Dataset.synthetic_rng(dom, data_size, rng)
    data = Dataset(raw_data_array, dom)
    numeric_features = data.domain.get_numeric_cols()

    stat_mod, _ = Marginals.get_all_kway_combinations(data.domain, 2, bins=[2])
    stat_mod.fit(data)

    stat_fn = stat_mod.get_stat_fn()
    stats1 = stat_fn(data.to_numpy())
    print(stats1)
    print('test_discrete() passed!')


if __name__ == "__main__":
    # test_mixed()
    # test_runtime()
    # test_cat_and_diff()
    test_mixed()
    # test_discrete()
    # test_row_answers()