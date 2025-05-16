import itertools
import jax
import jax.numpy as jnp
import chex
from genetic_sd.utils import Dataset, Domain
from genetic_sd.adaptive_statistics import AdaptiveStatisticState
from tqdm import tqdm
import numpy as np


class Marginals(AdaptiveStatisticState):

    def __init__(self, domain: Domain, kway_combinations: list, k: int, tree_query_depth: int = 3,
                 float_granularity: float = 0.01,
                 num_cols_thresholds: dict = None):
        """

        :param domain: This represents the data schema
        :param kway_combinations: A list representing the k-way relations that this module will compute
        :param k:
        :param bins:
        :param levels:
        """
        self.domain = domain
        self.kway_combinations = kway_combinations
        self.k = k
        self.tree_query_depth = tree_query_depth
        self.float_granularity = float_granularity
        self.workload_positions = []
        self.workload_sensitivity = []
        self.numerical_col_bins = num_cols_thresholds

        self.set_up_stats()

    def __str__(self):
        return f'Marginals'

    def get_num_workloads(self):
        return len(self.workload_positions)

    def _get_workload_positions(self, workload_id: int = None) -> tuple:
        return self.workload_positions[workload_id]

    def is_workload_numeric(self, cols):
        for c in cols:
            if c in self.domain.get_continuous_cols() or c in self.domain.get_ordinal_cols():
                return True
        return False

    def set_up_stats(self):
        """
        Build workloads: By definition each workload consist of a vector statistic function with l1 sensitivity = 1.
        :return:
        """

        if self.numerical_col_bins is None:
            self.numerical_col_bins = {}
        # ---------------------------------- #
        # Add default float column bins
        # ---------------------------------- #
        for num_col in  self.domain.get_continuous_cols():
            # Check if this column already defines bin edges
            if num_col not in self.numerical_col_bins:
                min_value, max_value = self.domain.range(num_col)
                thres = np.linspace(min_value, max_value, 2 ** self.tree_query_depth)
                self.numerical_col_bins[num_col] = thres
        for num_col in  self.domain.get_ordinal_cols():
            # Check if this column already defines bin edges
            if num_col not in self.numerical_col_bins:
                min_value, max_value = self.domain.range(num_col)
                size = max_value - min_value + 1
                thres = np.linspace(min_value, max_value, size)
                self.numerical_col_bins[num_col] =thres

        queries = []
        self.workload_positions = [] # Index
        self.workload_sensitivity = []
        for marginal in tqdm(self.kway_combinations, desc='Setting up Marginals.'):
            assert len(marginal) == self.k
            indices = self.domain.get_attribute_indices(marginal)
            levels = self.tree_query_depth if self.is_workload_numeric(marginal) else 1
            # start_pos = len(queries)
            for level in range(levels):
                start_pos = len(queries)
                intervals = []
                for att in marginal:
                    size = self.domain.size(att)
                    if self.domain.is_categorical(att):
                        upper = np.linspace(0, size, num=size+1)[1:]
                        lower = np.linspace(0, size, num=size+1)[:-1]
                        interval = list(np.vstack((upper, lower)).T - 0.1)
                        intervals.append(interval)
                    # elif self.domain.is_ordinal(att):
                    #     ord_bins = (size + 1) // (2**level)
                    #     ord_bins = max(ord_bins, 3)  # There must be at least 3 bins
                    #     upper = np.linspace(0, size, num=ord_bins)[1:]
                    #     lower = np.linspace(0, size, num=ord_bins)[:-1]
                    #     interval = list(np.vstack((upper, lower)).T - 0.0001)
                    #     intervals.append(interval)
                    else:
                        # ---------------------- #
                        # Process numeric queries
                        # ---------------------- #
                        bins_att = self.numerical_col_bins[att]
                        num_bins = bins_att.shape[0]
                        min_val, max_val = self.domain.range(att)
                        part = 2**level
                        if num_bins // part < 3:
                            part = num_bins // 2
                        upper = bins_att[part::part]
                        lower = bins_att[:-part:part]
                        upper[-1] = max_val
                        lower[0] = min_val
                        interval = list(np.vstack((upper, lower)).T)
                        intervals.append(interval)

                for v in itertools.product(*intervals):
                    v_arr = np.array(v)
                    upper = v_arr.flatten()[::2]
                    lower = v_arr.flatten()[1::2]
                    q = np.concatenate((indices, upper, lower))
                    queries.append(q)  # (i1, i2), ((a1, a2), (b1, b2))

                # Workload ends here.
                end_pos = len(queries)
                self.workload_positions.append((start_pos, end_pos))
                self.workload_sensitivity.append(jnp.sqrt(2))
                # print(f'Marginal ', marginal, f' (Level={level}).  Num queries = ', end_pos - start_pos)

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


    @staticmethod
    def get_all_kway_combinations(domain: Domain, k: int, tree_query_depth: int = 3, max_workload_size: int =None,
                                  include_feature: str=None,
                                  num_cols_thresholds: dict = None):
        kway_combinations = [list(idx) for idx in itertools.combinations(domain.attrs, k)]

        new_kway_comb = []
        for comb in kway_combinations:
            # --------------------------------------------- #
            # Compute the number of queries in this workload
            # --------------------------------------------- #
            workload_size = 1

            for att in comb:
                sz = domain.size(att)
                if sz > 1:
                    workload_size = workload_size * sz
                else:
                    # Get the bin_edges for attribute att
                    has_bin_edges, bin_edges = domain.get_bin_edges(att)

                    if has_bin_edges:
                        workload_size = workload_size * len(bin_edges)
                    else:
                        workload_size = workload_size * 64

            if (max_workload_size is None) or workload_size < max_workload_size:
                if include_feature is None or include_feature in comb:
                    new_kway_comb.append(comb)
        kway_combinations = new_kway_comb

        return Marginals(domain, kway_combinations, k, tree_query_depth=tree_query_depth, num_cols_thresholds=num_cols_thresholds)

    @staticmethod
    def get_all_kway_mixed_combinations_v1(domain, k):
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

        return Marginals(domain, kway_combinations, k)








