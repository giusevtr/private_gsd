import itertools
import jax
import jax.numpy as jnp
import chex
from utils import Dataset, Domain
from stats import AdaptiveStatisticState
from tqdm import tqdm
import numpy as np


class Marginals(AdaptiveStatisticState):

    def __init__(self, domain, kway_combinations, k, bins=None, levels=3):
        self.domain = domain
        self.kway_combinations = kway_combinations
        self.k = k
        self.levels = levels
        self.workload_positions = []
        self.workload_sensitivity = []

        self.bins = bins if bins is not None else {}
        for num_col in domain.get_numerical_cols():
            if num_col not in self.bins:
                self.bins[num_col] = np.linspace(0, 1, 2 ** levels)

        self.set_up_stats()

    def __str__(self):
        return f'Marginals'

    def get_num_workloads(self):
        return len(self.workload_positions)

    def _get_workload_positions(self, workload_id: int = None) -> tuple:
        return self.workload_positions[workload_id]

    def is_workload_numeric(self, cols):
        for c in cols:
            if c in self.domain.get_numerical_cols() or c in self.domain.get_ordinal_cols():
                return True
        return False

    def set_up_stats(self):

        queries = []
        self.workload_positions = []
        for marginal in tqdm(self.kway_combinations, desc='Setting up Marginals.'):
            assert len(marginal) == self.k
            indices = self.domain.get_attribute_indices(marginal)

            levels = self.levels if self.is_workload_numeric(marginal) else 1
            start_pos = len(queries)
            for level in range(levels):
                intervals = []
                for att in marginal:
                    size = self.domain.size(att)
                    if self.domain.type(att) == 'categorical':
                        upper = np.linspace(0, size, num=size+1)[1:]
                        lower = np.linspace(0, size, num=size+1)[:-1]
                        interval = list(np.vstack((upper, lower)).T - 0.1)
                        intervals.append(interval)
                    elif self.domain.type(att) == 'ordinal':
                        ord_bins = (size + 1) // (2**level)
                        ord_bins = max(ord_bins, 3)  # There must be at least 3 bins
                        upper = np.linspace(0, size, num=ord_bins)[1:]
                        lower = np.linspace(0, size, num=ord_bins)[:-1]
                        interval = list(np.vstack((upper, lower)).T - 0.0001)
                        intervals.append(interval)
                    else:
                        bins_att = self.bins[att]
                        num_bins = bins_att.shape[0]

                        part = 2**level
                        if num_bins // part < 3:
                            part = num_bins // 2
                        upper = bins_att[part::part]
                        lower = bins_att[:-part:part]
                        interval = list(np.vstack((upper, lower)).T)
                        intervals.append(interval)

                for v in itertools.product(*intervals):
                    v_arr = np.array(v)
                    upper = v_arr.flatten()[::2]
                    lower = v_arr.flatten()[1::2]
                    q = np.concatenate((indices, upper, lower))
                    queries.append(q)  # (i1, i2), ((a1, a2), (b1, b2))
            end_pos = len(queries)
            self.workload_positions.append((start_pos, end_pos))
            self.workload_sensitivity.append(jnp.sqrt(2 * levels))

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

    def _get_stat_fn(self, query_ids):
        def answer_fn(x_row: chex.Array, query_single: chex.Array):
            I = query_single[:self.k].astype(int)
            U = query_single[self.k:2 * self.k]
            L = query_single[2 * self.k:3 * self.k]
            t1 = (x_row[I] < U).astype(int)
            t2 = (x_row[I] >= L).astype(int)
            t3 = jnp.prod(jnp.array([t1, t2]), axis=0)
            answers = jnp.prod(t3)
            return answers

        these_queries = self.queries[query_ids]
        temp_stat_fn = jax.vmap(answer_fn, in_axes=(None, 0))

        def scan_fun(carry, x):
            return carry + temp_stat_fn(x, these_queries), None
        def stat_fn(X):
            out = jax.eval_shape(temp_stat_fn, X[0], these_queries)
            stats = jax.lax.scan(scan_fun, jnp.zeros(out.shape, out.dtype), X)[0]
            return stats / X.shape[0]
        return stat_fn

    def _get_numpy_workload_fn(self, workload_ids):
        pass
    @staticmethod
    def get_kway_categorical(domain: Domain, k):
        kway_combinations = [list(idx) for idx in itertools.combinations(domain.get_categorical_cols(), k)]
        return Marginals(domain, kway_combinations, k, bins=None)

    @staticmethod
    def get_all_kway_combinations(domain, k, bin_edges=None, levels=3, max_workload_size=None,
                                  include_feature=None):
        kway_combinations = [list(idx) for idx in itertools.combinations(domain.attrs, k)]

        new_kway_comb = []
        for comb in kway_combinations:
            workload_size = 1
            for att in comb:
                sz = domain.size(att)
                if sz > 1:
                    workload_size = workload_size * sz
                else:
                    if bin_edges is not None and att in bin_edges:
                        workload_size = workload_size * len(bin_edges[att])
                    else:
                        workload_size = workload_size * 64


            if (max_workload_size is None) or workload_size < max_workload_size:
                if include_feature is None or include_feature in comb:
                    new_kway_comb.append(comb)
        kway_combinations = new_kway_comb

        return Marginals(domain, kway_combinations, k, bins=bin_edges, levels=levels)

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

        return Marginals(domain, kway_combinations, k, bins=bins)
