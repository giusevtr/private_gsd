import chex
import jax.numpy as jnp
import jax
import numpy as np

from typing import Callable
from utils import Dataset, Domain
from tqdm import tqdm

from stats import  AdaptiveStatisticState
class ChainedStatistics(AdaptiveStatisticState):
    all_workloads: list
    selected_workloads: list
    # adaptive_rounds_count: int
    domain: Domain

    def __init__(self, stat_modules: list):
        self.stat_modules = stat_modules

    # def fit(self, data: Dataset):
    #     self.N = len(data.df)
    #     self.selected_workloads = []
    #     for stat_mod in self.stat_modules:
    #         stat_mod: AdaptiveStatisticState
    #         stat_mod.fit(data)
    #         self.selected_workloads.append([])
    def fit(self, data: Dataset):
        X = data.to_numpy()
        self.domain = data.domain
        num_attrs = len(self.domain.attrs)
        self.N = X.shape[0]
        self.all_workloads = []
        self.selected_workloads = []
        # self.real_stats = []
        for stat_id in range(len(self.stat_modules)):
            stat_mod: AdaptiveStatisticState
            stat_mod = self.stat_modules[stat_id]

            num_workloads = stat_mod.get_num_workloads()
            workload_fn = stat_mod._get_workload_fn()
            all_stats = workload_fn(X)

            self.all_workloads.append([])
            for i in tqdm(range(num_workloads), f'Fitting workloads. Data has {self.N} rows and {num_attrs} attributes.'):
                workload_fn = stat_mod._get_workload_fn(workload_ids=[i])
                # stats = workload_fn(X)
                wrk_a, wrk_b = stat_mod._get_workload_positions(i)
                stats = all_stats[wrk_a:wrk_b]
                self.all_workloads[stat_id].append((i, workload_fn, stats))
            self.selected_workloads.append([])

        self.all_statistics_fn = self._get_workload_fn()

    # def get_statistics_ids(self):
    #     return [tup[0] for tup in self.selected_workloads]

    def __get_selected_workload_ids(self, stat_id: int):
        return [tup[0] for tup in self.selected_workloads[stat_id]]

    def get_all_true_statistics(self):

        chained_stats = []
        for stat_id in range(len(self.stat_modules)):
            chained_stats.append(jnp.concatenate([tup[2] for tup in self.all_workloads[stat_id]]))
        # return jnp.concatenate([tup[2] for tup in stat_mod.all_workloads for stat_mod in self.stat_modules])
        return jnp.concatenate(chained_stats)

    def get_all_statistics_fn(self):
        return self._get_workload_fn()

    def get_selected_noised_statistics(self):
        selected_chained_stats = []
        for stat_id in range(len(self.stat_modules)):
            stat_mod = self.stat_modules[stat_id]
            temp = jnp.concatenate([selected[2] for selected in self.selected_workloads[stat_id]])
            selected_chained_stats.append(temp)

        return jnp.concatenate(selected_chained_stats)

    def get_selected_statistics_without_noise(self):
        # return jnp.concatenate([selected[3] for selected in self.selected_workloads])
        selected_chained_stats = []
        for stat_id in range(len(self.stat_modules)):
            stat_mod = self.stat_modules[stat_id]
            temp = jnp.concatenate([selected[3] for selected in self.selected_workloads[stat_id]])
            selected_chained_stats.append(temp)
        return jnp.concatenate(selected_chained_stats)

    def get_selected_statistics_fn(self):
        workload_fn_list = []
        for stat_id in range(len(self.stat_modules)):
            stat_mod = self.stat_modules[stat_id]
            workload_ids = self.__get_selected_workload_ids(stat_id)
            workload_fn_list.append(stat_mod._get_workload_fn(workload_ids))

        def chained_workload(X):
            return jnp.concatenate([fn(X) for fn in workload_fn_list], axis=0)

        return chained_workload

    def get_domain(self):
        return self.domain

    def get_num_workloads(self) -> int:
        s = 0
        for stat_mod in self.stat_modules:
            stat_mod: AdaptiveStatisticState
            s += stat_mod.get_num_workloads()
        return s

    def _get_workload_fn(self, workload_ids: list = None) -> Callable:
        workload_fn_list = []
        for stat_mod in self.stat_modules:
            stat_mod: AdaptiveStatisticState
            workload_fn_list.append(stat_mod._get_workload_fn())
        def chained_workload(X):
            return jnp.concatenate([fn(X) for fn in workload_fn_list], axis=0)
        return chained_workload

    def _get_workload_sensitivity(self, workload_id: int = None, N: int = None) -> float:
        pass

    def __add_stats(self, stat_id, workload_id, workload_fn, noised_workload_statistics, true_workload_statistics):
        stat_id = int(stat_id)
        workload_id = int(workload_id)
        self.selected_workloads[stat_id].append((workload_id, workload_fn, noised_workload_statistics, true_workload_statistics))

    def private_measure_all_statistics(self, key: chex.PRNGKey, rho: float):
        # self.adaptive_rounds_count += 1
        self.selected_workloads = []
        for stat_id in range(len(self.stat_modules)):
            self.selected_workloads.append([])

        m = self.get_num_workloads()
        rho_per_marginal = rho / m

        for stat_id in range(len(self.stat_modules)):
            stat_mod = self.stat_modules[stat_id]
            for workload_id in range(stat_mod.get_num_workloads()):
                key, key_gaussian = jax.random.split(key, 2)
                # _, workload_fn, stats = stat_mod.all_workloads[workload_id]
                _, workload_fn, stats = self.all_workloads[stat_id][workload_id]
                sensitivity = stat_mod._get_workload_sensitivity(workload_id, self.N)


                sigma_gaussian = float(np.sqrt(sensitivity ** 2 / (2 * rho_per_marginal)))
                gau_noise = jax.random.normal(key_gaussian, shape=stats.shape) * sigma_gaussian
                selected_noised_stat = jnp.clip(stats + gau_noise, 0, 1)
                self.__add_stats(stat_id, workload_id, workload_fn, selected_noised_stat, stats)

    # def private_select_measure_statistic(self, key: chex.PRNGKey, rho_per_round: float, sync_data_mat: chex.Array,
    #                                      sample_num=1):
    #     rho_per_round = rho_per_round / 2
    #     # STAT = self.STAT_MODULE
    #     selected_stats = jnp.array(self.get_statistics_ids()).astype(int)
    #
    #     errors = self.get_sync_data_errors(sync_data_mat)
    #     # max_sensitivity = max(self.STAT_MODULE.sensitivity)
    #     key, key_g = jax.random.split(key, 2)
    #     rs = np.random.RandomState(key_g)
    #     # Computing Gumbel noise scale based on: https://differentialprivacy.org/one-shot-top-k/
    #
    #     noise = rs.gumbel(scale=(np.sqrt(sample_num) / (np.sqrt(2 * rho_per_round) * self.N)), size=errors.shape)
    #     # noise = rs.gumbel(scale=(np.sqrt(sample_num) / (np.sqrt(2 * rho_per_round) * STAT.N)), size=errors.shape)
    #     noise = jnp.array(noise)
    #     errors_noise = errors + noise
    #     errors_noise = errors_noise.at[selected_stats].set(-100000)
    #     top_k_indices = (-errors_noise).argsort()[:sample_num]
    #     for worse_index in top_k_indices:
    #         gaussian_rho_per_round = rho_per_round / sample_num
    #         key, key_gaussian = jax.random.split(key, 2)
    #
    #         _, selected_workload_fn, selected_true_stat = self.all_workloads[worse_index]
    #         # selected_true_stat = STAT.get_true_stat([worse_index])
    #         sensitivity = self._get_workload_sensitivity(worse_index, self.N)
    #         sigma_gaussian = float(np.sqrt(sensitivity ** 2 / (2 * gaussian_rho_per_round)))
    #         gau_noise = jax.random.normal(key_gaussian, shape=selected_true_stat.shape) * sigma_gaussian
    #         selected_priv_stat = jnp.clip(selected_true_stat + gau_noise, 0, 1)
    #         self.add_stats(worse_index, selected_workload_fn, selected_priv_stat, selected_true_stat)


    def get_sync_data_errors(self, X):
        max_errors = []
        for stat_id in range(len(self.stat_modules)):
            stat_mod = self.stat_modules[stat_id]
            for workload_id, workload_fn, statistics in stat_mod.all_workloads:
                workload_error = jnp.abs(statistics - workload_fn(X))
                max_errors.append(workload_error.max())
        # for i in range(num_workloads):
        #     fn = self.marginals_fn_jit[i]
        #     error = jnp.abs(self.true_stats[i] - fn(X))
        #     max_errors.append(error.max())
        return jnp.array(max_errors)



def exponential_mechanism(key: jnp.ndarray, scores: jnp.ndarray, eps0: float, sensitivity: float):
    dist = jax.nn.softmax(2 * eps0 * scores / (2 * sensitivity))
    cumulative_dist = jnp.cumsum(dist)
    max_query_idx = jnp.searchsorted(cumulative_dist, jax.random.uniform(key, shape=(1,)))
    return max_query_idx[0]



if __name__ == "__main__":
    from stats import Marginals
    from toy_datasets.classification import  get_classification

    data = get_classification()


    marginal_module1, _ = Marginals.get_all_kway_combinations(data.domain, k=1, bins=[2, 4])
    marginal_module2, _ = Marginals.get_all_kway_combinations(data.domain, k=2, bins=[2, 4])

    marginal_module1.fit(data)
    marginal_module2.fit(data)
    alls_stats0 = jnp.concatenate([marginal_module1.get_all_true_statistics(),
                                   marginal_module2.get_all_true_statistics()])


    chained_module = ChainedStatistics([marginal_module1, marginal_module2])
    chained_module.fit(data)

    all_stats1 = chained_module.get_all_true_statistics()

    all_stats2 = chained_module.all_statistics_fn(data.to_numpy())
    stat_fn = chained_module.get_all_statistics_fn()
    all_stats3 = stat_fn(data.to_numpy())

    print(alls_stats0)
    print(all_stats1)
    print(all_stats2)
    print(all_stats3)



