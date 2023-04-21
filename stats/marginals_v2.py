import itertools
# import jax
# import jax.numpy as jnp
import chex
from utils import Dataset, Domain
from stats import AdaptiveStatisticState
from tqdm import tqdm
import numpy as np
from typing import List
# from jax._src.typing import Array
import math

from flax import struct
# @struct.dataclass
# class Workload:
#     index: chex.Array
#     remove_row: chex.Array
#     add_row: chex.Array

class MarginalsV2(AdaptiveStatisticState):

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
                self.bins[num_col] = np.linspace(0, 1, 64)

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
        pass

    def _get_workload_sensitivity(self, workload_id: int = None, N: int = None) -> float:
        return self.workload_sensitivity[workload_id] / N

    def _get_dataset_statistics_fn(self, workload_ids=None, jitted: bool = False):
        workload_fn = self._get_workload_fn(workload_ids)

        def data_fn(data: Dataset):
            X = data.to_numpy_np()
            return workload_fn(X)
        return data_fn

    def _get_workload_fn(self, workload_ids=None):

        answer_fn = self._get_row_fn(workload_ids)

        def stat_fn(X):
            stats = answer_fn(X)
            return stats / X.shape[0]
        return stat_fn

    def _get_row_fn(self, workload_ids=None):
        """
        Returns marginals function and sensitivity
        :return:
        """
        dim = len(self.domain.attrs)


        col_bins = []
        for _ in self.domain.attrs:
            col_bins.append([])

        col_levels = np.zeros(dim)
        for att in self.domain.attrs:
            index = self.domain.get_attribute_indices(att)[0]
            size = self.domain.size(att)
            if self.domain.type(att) == 'categorical':
                bins = np.linspace(0, size-1, num=size)
                col_bins[index].append(bins)
            elif self.domain.type(att) == 'ordinal':
                for level in range(self.levels):
                    ord_bins = (size) // (2 ** level)
                    ord_bins = max(ord_bins, 3)  # There must be at least 3 bins
                    bins = np.linspace(0, size-1, num=ord_bins)
                    col_bins[index].append(bins)
            else:
                bins_att = self.bins[att]
                num_bins = bins_att.shape[0]
                for level in range(self.levels):
                    part = 2 ** level
                    if num_bins // part < 3:
                        part = num_bins // 2
                    bins = bins_att[part::part]
                    col_bins[index].append(bins)

            # How many levels in this index.
            col_levels[index] = len(col_bins[index])

        col_levels = np.array(col_levels)


        if workload_ids is None:
            combinations = self.kway_combinations
        else:
            combinations = [self.kway_combinations[i] for i in workload_ids]

        workload_indices= []
        workload_bin_edges= []
        for marginal in combinations:
            assert len(marginal) == self.k
            indices = self.domain.get_attribute_indices(marginal)

            max_levels = int(col_levels[indices].max())
            for level in range(max_levels):
                workload_indices.append(np.array(indices))
                bin_edges_temp = []
                for i in indices:
                    this_level = min(level, len(col_bins[i]))
                    bin_edges_temp.append(col_bins[i][this_level])
                workload_bin_edges.append(bin_edges_temp)

        def answer_fn(X: chex.Array):

            histograms= []

            for indices, all_bin_edges in zip(workload_indices, workload_bin_edges):
                # x_row = x_row_all[indices]
                # x_row = X[]
                bin_idx_by_dim = []
                bin_edges_by_dim = []

                for i, index in enumerate(indices):
                    bin_edges = all_bin_edges[i]
                    bin_idx = np.searchsorted(bin_edges, X[:, index], side='left')
                    bin_idx_by_dim.append(bin_idx)
                    bin_edges_by_dim.append(bin_edges)

                nbins = tuple(len(bin_edges) for bin_edges in bin_edges_by_dim)
                xy = np.ravel_multi_index(tuple(bin_idx_by_dim), nbins, mode='clip')
                hist = np.bincount(xy, None, minlength=math.prod(nbins))
                histograms.append(hist.ravel())
            return np.concatenate(histograms)

        return answer_fn


    @staticmethod
    def get_kway_categorical(domain: Domain, k):
        kway_combinations = [list(idx) for idx in itertools.combinations(domain.get_categorical_cols(), k)]
        return MarginalsV2(domain, kway_combinations, k, bins=None)

    @staticmethod
    def get_all_kway_combinations(domain, k, bins=None, levels=3, max_workload_size=None):
        kway_combinations = [list(idx) for idx in itertools.combinations(domain.attrs, k)]

        if max_workload_size is not None:
            new_kway_comb = []
            for comb in kway_combinations:
                workload_size = 1
                for att in comb:
                    sz = domain.size(att)
                    if sz > 1:
                        workload_size = workload_size * sz
                    else:
                        if att in bins:
                            workload_size = workload_size * len(bins[att])
                        else:
                            workload_size = workload_size * 64


                if workload_size < max_workload_size:
                    # kway_combinations.remove(comb)
                    new_kway_comb.append(comb)
            kway_combinations = new_kway_comb

        return MarginalsV2(domain, kway_combinations, k, bins=bins, levels=levels)

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

        return MarginalsV2(domain, kway_combinations, k, bins=bins)






def test_marginals():

    domain = Domain({
        'A': {'type': 'categorical', 'size': 2},
        'B': {'type': 'categorical', 'size': 2},
        'C': {'type': 'categorical', 'size': 3},
    })
    data = Dataset.synthetic(domain, N=19, seed=0)

    # module = MarginalsV2.get_kway_categorical(domain, k=2)
    module = MarginalsV2(domain, k=2, kway_combinations=[('A', 'B'), ('B', 'C')])

    stat_fn = module._get_row_fn()
    # stat_fn_jit = jax.jit(stat_fn)

    print(stat_fn(np.array([[0, 0, 0]])))


if __name__ == "__main__":

    test_marginals()




