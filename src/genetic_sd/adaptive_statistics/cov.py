import itertools
import jax
import jax.numpy as jnp
import chex
from genetic_sd.utils import Dataset, Domain
from genetic_sd.adaptive_statistics import AdaptiveStatisticState
from tqdm import tqdm
import numpy as np


class Covariance(AdaptiveStatisticState):

    def __init__(self, domain: Domain,
                 numerical_cols_meta_data: dict,  #
                 num_intervals: int = 64,
                 ):
        """

        :param domain: This represents the data schema
        :param kway_combinations: A list representing the k-way relations that this module will compute
        :param k:
        :param bins:
        :param levels:
        """
        self.domain = domain
        # self.k = k
        # self.tree_query_depth = tree_query_depth
        self.workload_positions = []
        self.workload_sensitivity = []
        # self.numerical_col_bins = num_cols_thresholds
        self.numerical_cols_meta_data = numerical_cols_meta_data
        self.num_intervals = num_intervals

        self.set_up_stats()

    def __str__(self):
        return f'Covariance'

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
        # ---------------------------------- #
        # Add default float column bins
        # ---------------------------------- #

        queries = []
        self.workload_positions = [] # Index
        self.workload_sensitivity = []


        col_pairs = itertools.combinations(self.domain.get_continuous_cols() + self.domain.get_ordinal_cols(), 2)


        for marginal in tqdm(col_pairs, desc='Setting up Covariance.'):
            indices = self.domain.get_attribute_indices(marginal)

            col1, col2 = marginal

            start_pos = len(queries)

            intervals = np.linspace(-2, 2, self.num_intervals)

            cols_mean = np.array([self.numerical_cols_meta_data[col1]['mean'], self.numerical_cols_meta_data[col2]['mean']])
            cols_std = np.array([self.numerical_cols_meta_data[col1]['std'], self.numerical_cols_meta_data[col2]['std']])
            for i in range(self.num_intervals-1):
                lower = intervals[i]
                upper = intervals[i+1]
                # v_arr = np.array(v)
                # upper = v_arr.flatten()[::2]
                # lower = v_arr.flatten()[1::2]

                q = np.concatenate((indices, cols_mean, cols_std, [lower], [upper]))
                queries.append(q)  # (i1, i2), ((a1, a2), (b1, b2))

            # Workload ends here.
            end_pos = len(queries)
            self.workload_positions.append((start_pos, end_pos))
            self.workload_sensitivity.append(jnp.sqrt(2))
            # print(f'Marginal ', marginal, f' (Level={level}).  Num queries = ', end_pos - start_pos)

        self.queries = jnp.array(queries)


    def _get_stat_fn(self, query_ids):
        def answer_fn(x_row: chex.Array, query_single: chex.Array):
            I = query_single[:2].astype(int)
            col_means = query_single[2:4]
            col_std = query_single[4:6]

            # Normalize values
            x_normed = (x_row[I] - col_means) / col_std

            # Compute product
            x_prod = jnp.prod(x_normed)

            # Check if product is in the interval
            L = query_single[6]
            U = query_single[7]
            t1 = (x_prod < U).astype(int)
            t2 = (x_prod >= L).astype(int)
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










