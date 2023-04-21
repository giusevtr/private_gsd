import itertools
import jax
import jax.numpy as jnp
import chex
from utils import Dataset, Domain
from stats import AdaptiveStatisticState
from tqdm import tqdm
import numpy as np
from typing import List
from jax._src.typing import Array
import math

from flax import struct
# @struct.dataclass
# class Workload:
#     index: chex.Array
#     remove_row: chex.Array
#     add_row: chex.Array

class MarginalsV3(AdaptiveStatisticState):

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

        dim = len(self.domain.attrs)

        if workload_ids is None:
            combinations = self.kway_combinations
        else:
            combinations = [self.kway_combinations[i] for i in workload_ids]

        col_bins: List[List[Array]] = []
        for _ in self.domain.attrs:
            col_bins.append([])

        col_sizes = np.zeros(dim)
        col_ranges = [None for _ in range(dim)]

        for att in self.domain.attrs:
            index = self.domain.get_attribute_indices(att)[0]
            size = self.domain.size(att)
            if self.domain.type(att) == 'categorical':
                col_sizes[index] = size + 1
                col_ranges[index] = np.array([0, size+1])
            elif self.domain.type(att) == 'ordinal':
                col_sizes[index] = size + 1
                col_ranges[index] = np.array([0, size+1])
            else:
                col_sizes[index] = 64
                col_ranges[index] = np.array([0, 1])

        col_sizes = np.array(col_sizes)
        col_ranges = np.array(col_ranges)

        workload_indices: List[Array] = []
        for marginal in combinations:
            assert len(marginal) == self.k
            indices = self.domain.get_attribute_indices(marginal)
            workload_indices.append(indices)

        def stat_fn(X):
            histograms: List[Array] = []

            for indices in workload_indices:
                X_proj = X[:, indices]
                bins = col_sizes[indices].astype(int)
                ranges = col_ranges[indices]
                hist = np.histogramdd(X_proj, bins=bins, range=ranges)[0]
                histograms.append(hist.ravel())
            return jnp.concatenate(histograms) / X.shape[0]

        return stat_fn

    @staticmethod
    def get_kway_categorical(domain: Domain, k):
        kway_combinations = [list(idx) for idx in itertools.combinations(domain.get_categorical_cols(), k)]
        return MarginalsV3(domain, kway_combinations, k, bins=None)

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

        return MarginalsV3(domain, kway_combinations, k, bins=bins, levels=levels)

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

        return MarginalsV3(domain, kway_combinations, k, bins=bins)






def test_marginals():

    domain = Domain({
        'A': {'type': 'categorical', 'size': 2},
        'B': {'type': 'categorical', 'size': 2},
        'C': {'type': 'categorical', 'size': 3},
    })
    data = Dataset.synthetic(domain, N=19, seed=0)

    # module = MarginalsV2.get_kway_categorical(domain, k=2)
    module = MarginalsV3(domain, k=2, kway_combinations=[('A', 'B'), ('B', 'C')])

    stat_fn = module._get_row_fn()
    stat_fn_jit = jax.jit(stat_fn)

    print(stat_fn(jnp.array([0, 0, 0])))
    print(stat_fn(jnp.array([0, 0, 1])))
    print(stat_fn(jnp.array([0, 0, 2])))
    print(stat_fn(jnp.array([0, 1, 0])))
    print(stat_fn(jnp.array([0, 1, 1])))
    print(stat_fn(jnp.array([0, 1, 2])))

if __name__ == "__main__":

    test_marginals()




