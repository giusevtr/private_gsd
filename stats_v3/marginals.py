import itertools
import jax
import jax.numpy as jnp
from utils import Dataset, Domain
from utils.utils_data import Domain
import numpy as np


class PrivateMarginalsState:
    def __init__(self):
        self.true_stats = []
        self.priv_stats = []
        self.priv_marginals_fn = []
        self.priv_diff_marginals_fn = []
        self.selected_marginals = []

    def get_true_stats(self):
        return jnp.concatenate(self.true_stats)

    def get_priv_stats(self):
        return jnp.concatenate(self.priv_stats)

    def get_stats(self, X):
        return jnp.concatenate([fn(X) for fn in self.priv_marginals_fn])

    def get_diff_stats(self, X):
        return jnp.concatenate([diff_fn(X) for diff_fn in self.priv_diff_marginals_fn])

    def true_loss_inf(self, X):
        true_stats_concat = jnp.concatenate(self.true_stats)
        sync_stats_concat = self.get_stats(X)
        return jnp.abs(true_stats_concat - sync_stats_concat).max()

    def true_loss_l2(self, X):
        true_stats_concat = jnp.concatenate(self.true_stats)
        sync_stats_concat = self.get_stats(X)
        return jnp.linalg.norm(true_stats_concat - sync_stats_concat, ord=2) / true_stats_concat.shape[0]

    def priv_loss_inf(self, X):
        priv_stats_concat = jnp.concatenate(self.priv_stats)
        sync_stats_concat = self.get_stats(X)
        return jnp.abs(priv_stats_concat - sync_stats_concat).max()

    def priv_loss_l2(self, X):
        priv_stats_concat = jnp.concatenate(self.priv_stats)
        sync_stats_concat = self.get_stats(X)
        return jnp.linalg.norm(priv_stats_concat - sync_stats_concat, ord=2) / priv_stats_concat.shape[0]

    def priv_diff_loss_inf(self, X_oh):
        priv_stats_concat = jnp.concatenate(self.priv_stats)
        sync_stats_concat = self.get_diff_stats(X_oh)
        return jnp.abs(priv_stats_concat - sync_stats_concat).max()

    def priv_diff_loss_l2(self, X_oh):
        priv_stats_concat = jnp.concatenate(self.priv_stats)
        sync_stats_concat = self.get_diff_stats(X_oh)
        return jnp.linalg.norm(priv_stats_concat - sync_stats_concat, ord=2) ** 2 / priv_stats_concat

class Marginals:
    true_stats: list = None
    marginals_fn: list
    domain: Domain
    N: int = None  # Dataset size

    def __init__(self, domain, kway_combinations, bins=30, name='Marginals'):
        self.domain = domain
        self.kway_combinations = kway_combinations
        self.bins = bins

    @staticmethod
    def get_all_kway_combinations(domain, k, bins=30):
        kway_combinations = [list(idx) for idx in itertools.combinations(domain.attrs, k)]
        return Marginals(domain, kway_combinations, bins=bins, name=f'{k}-way Marginals')

    def fit(self, data: Dataset):
        self.true_stats = []
        self.marginals_fn = []
        self.diff_marginals_fn = []

        X = data.to_numpy()
        self.N = X.shape[0]
        self.domain = data.domain

        for cols in self.kway_combinations:
            sizes = []
            for col in cols:
                sizes.append(self.domain.size(col))
            indices = self.domain.get_attribute_indices(cols)
            fn = self.get_marginal_stats_fn_helper(indices, sizes)
            self.true_stats.append(fn(X))
            self.marginals_fn.append(jax.jit(fn))
            diff_fn = self.get_differentiable_stats_fn_helper(cols)
            self.diff_marginals_fn.append(jax.jit(diff_fn))

    def get_num_queries(self):
        return len(self.kway_combinations)

    def get_dataset_size(self):
        return self.N

    def get_true_stats(self):
        assert self.true_stats is not None, "Error: must call the fit function"
        return jnp.concatenate(self.true_stats)


    def get_stats(self, data:Dataset, indices: list = None):
        X = data.to_numpy()
        stats = []
        I = indices if indices is not None else list(range(self.get_num_queries()))
        for i in I:
            fn = self.marginals_fn[i]
            stats.append(fn(X))
        return jnp.concatenate(stats)

    def get_diff_stats(self, data:Dataset, indices: list = None):
        X = data.to_onehot()
        stats = []
        I = indices if indices is not None else list(range(self.get_num_queries()))
        for i in I:
            fn = self.diff_marginals_fn[i]
            stats.append(fn(X))
        return jnp.concatenate(stats)


    def get_sync_data_errors(self, X):
        assert self.true_stats is not None, "Error: must call the fit function"
        max_errors = []
        for i in range(len(self.kway_combinations)):
            fn = self.marginals_fn[i]
            error = jnp.abs(self.true_stats[i] - fn(X))
            max_errors.append(error.max())
        return jnp.array(max_errors)


    def get_sensitivity(self):
        return jnp.sqrt(2) / self.N

    def get_marginal_stats_fn_helper(self, idx, sizes):

        cat_idx = self.domain.get_attribute_indices(self.domain.get_categorical_cols())
        sizes_jnp = [jnp.arange(s + 1).astype(float) if i in cat_idx else jnp.linspace(0, 1, self.bins) for i, s in zip(idx, sizes)]
        def stat_fn(X):
            # X_proj = (X[:, idx]).astype(int)
            X_proj = X[:, idx]
            stat = jnp.histogramdd(X_proj, sizes_jnp)[0].flatten() / X.shape[0]
            return stat
        return stat_fn

    def get_differentiable_stats_fn_helper(self, kway_attributes):
        # For computing differentiable marginal queries
        queries = []
        indices = [self.domain.get_attribute_onehot_indices(att) for att in kway_attributes]
        for tup in itertools.product(*indices):
            queries.append(tup)
        queries = jnp.array(queries)
        # queries_split = jnp.array_split(self.queries, 10)

        # @jax.jit
        def stat_fn(X):
            return jnp.prod(X[:, queries], 2).sum(0) / X.shape[0]

        return stat_fn
    ##################################################
    ## Adaptive statistics
    ##################################################
    def private_select_measure_statistic(self, key, rho_per_round, X_sync, state: PrivateMarginalsState):
        rho_per_round = rho_per_round / 2

        key, key_em = jax.random.split(key, 2)
        errors = self.get_sync_data_errors(X_sync)

        # errors = errors.at[]  # Zero out selected errors
        worse_index = exponential_mechanism(key_em, errors, jnp.sqrt(2 * rho_per_round), self.get_sensitivity())

        key, key_gaussian = jax.random.split(key, 2)
        selected_true_stat = self.true_stats[worse_index]

        sensitivity = self.get_sensitivity()
        sigma_gaussian = float(np.sqrt(sensitivity ** 2 / (2 * rho_per_round)))
        gau_noise = jax.random.normal(key_gaussian, shape=selected_true_stat.shape) * sigma_gaussian

        # self.priv_stats.append(selected_true_stat + gau_noise)
        # self.priv_marginals_fn.append(self.marginals_fn[worse_index])
        # self.selected_marginals.append(worse_index)

        state.true_stats.append(selected_true_stat)
        state.priv_stats.append(selected_true_stat + gau_noise)
        state.priv_marginals_fn.append(self.marginals_fn[worse_index])
        state.priv_diff_marginals_fn.append(self.diff_marginals_fn[worse_index])
        # state.selected_marginals.append(worse_index)

        return state



def exponential_mechanism(key:jnp.ndarray, scores: jnp.ndarray, eps0: float, sensitivity: float):
    dist = jax.nn.softmax(2 * eps0 * scores / (2 * sensitivity))
    cumulative_dist = jnp.cumsum(dist)
    max_query_idx = jnp.searchsorted(cumulative_dist, jax.random.uniform(key, shape=(1,)))
    return max_query_idx[0]


######################################################################
## TEST
######################################################################

def test_mixed():

    import numpy as np
    import pandas as pd
    print('debug')
    cols = ['A', 'B', 'C']
    domain = Domain(cols, [2, 1, 1])

    A = pd.DataFrame(np.array([
        [0, 0.0, 0.1],
        [0, 0.2, 0.3],
        [0, 0.8, 0.9],
        [0, 0.1, 0.0],
    ]), columns=cols)
    data = Dataset(A, domain=domain)
    X = data.to_numpy()

    stat_mod = Marginals.get_all_kway_combinations(domain, 2, bins=3)
    stat_mod.fit(data)

    stats1 = stat_mod.get_true_stats()
    print(stats1.shape)
    print(stats1)


def test_cat_and_diff():

    import numpy as np
    import pandas as pd
    print('debug')
    cols = ['A', 'B', 'C']
    domain = Domain(cols, [2, 3, 2])

    A = pd.DataFrame(np.array([
        [0, 0, 1],
        [0, 2, 0],
        [0, 0, 0],
        [0, 1, 0],
    ]), columns=cols)
    data = Dataset(A, domain=domain)
    X = data.to_numpy()

    stat_mod = Marginals.get_all_kway_combinations(domain, 2, bins=3)
    stat_mod.fit(data)

    stats_true = stat_mod.get_true_stats()
    stats_diff = stat_mod.get_diff_stats(data)

    print(jnp.linalg.norm(stats_true - stats_diff, ord=1))

def test_runtime():
    import time
    DATA_SIZE = 10000
    cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    domain = Domain(cols, [16, 7, 10, 1, 10, 13,  1, 12, 8, 10, 6])
    data = Dataset.synthetic(domain, DATA_SIZE, 0)

    stime = time.time()
    stat_mod = Marginals.get_all_kway_combinations(domain, 3)
    print(f'create stat_module elapsed time = {time.time() - stime:.5f}')

    print(f'num queries = {stat_mod.get_num_queries()}')
    print(f'Fitting')
    stime = time.time()
    stat_mod.fit(data)
    etime = time.time() - stime
    print(f'fit elapsed time = {etime:.5f}')

    true_stats = stat_mod.get_true_stats()
    print(f'true_stats.shape = ', true_stats.shape)
    print(f'Testing sub stat module evaluation time.')
    print(f'Testing submodule with 5 workloads...')

    sync_data = Dataset.synthetic(domain, 500, 0)

    sync_X = sync_data.to_numpy()
    # stat_fn = jax.jit(sub_stats_moudule.get_stats_fn())
    stime = time.time()
    for it in range(0, 101):
        stat = stat_mod.get_stats(sync_X, indices=[0, 1, 2, 3, 4])
        stat.block_until_ready()
        # print(stat)
        del stat
        if it % 50 == 0:
            print(f'1) it={it:02}. time = {time.time()-stime:.5f}')
    print(f'first evaluate elapsed time = {time.time() - stime:.5f}')

    print(f'Testing submodule with one more workload...')
    stime = time.time()
    for it in range(0, 101):
        D_temp = stat_mod.get_stats(sync_X, indices=[0, 1, 2, 3, 4, 5])
        del D_temp
        if it % 50 == 0:
            print(f'2) it={it:02}. time = {time.time()-stime:.5f}')
    print(f'second evaluate elapsed time = {time.time() - stime:.5f}')




if __name__ == "__main__":
    # test_mixed()
    # test_runtime()
    test_cat_and_diff()