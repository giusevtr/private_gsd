import itertools
import jax
import jax.numpy as jnp
from utils import Dataset, Domain
from utils.utils_data import Domain
from stats.private_statistics import PrivateMarginalsState
import numpy as np
import chex


class Marginals:
    true_stats: list
    marginals_fn: list
    marginals_fn_jit: list
    diff_marginals_fn: list
    diff_marginals_fn_jit: list
    sensitivity: list

    def __init__(self, domain, kway_combinations, bins=(32,)):
        self.domain = domain
        self.kway_combinations = kway_combinations
        self.bins = list(bins)

        # Check that domain constains real-valued features
        self.IS_REAL_VALUED = len(domain.get_numeric_cols()) > 0

    def __str__(self):
        return f'Marginals'

    @staticmethod
    def get_all_kway_combinations(domain, k, bins=(32,)):
        kway_combinations = [list(idx) for idx in itertools.combinations(domain.attrs, k)]
        return Marginals(domain, kway_combinations, bins=bins)

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

        return Marginals(domain, kway_combinations, bins=bins), kway_combinations

    @staticmethod
    def get_all_kway_mixed_combinations(domain, k_disc, k_real, bins=(32,)):
        num_numeric_feats = len(domain.get_numeric_cols())
        k_real = min(num_numeric_feats, k_real)

        kway_combinations = []

        K = k_disc + k_real
        for cols in itertools.combinations(domain.attrs, K):
            count_disc = 0
            count_real = 0
            for c in list(cols):
                if c in domain.get_numeric_cols():
                    count_real += 1
                else:
                    count_disc += 1
            if count_disc == k_disc and count_real == k_real:
                kway_combinations.append(list(cols))

        return Marginals(domain, kway_combinations, bins=bins), kway_combinations

    def fit(self, data: Dataset):
        self.true_stats = []
        self.marginals_fn = []
        self.marginals_fn_jit = []
        self.diff_marginals_fn = []
        self.diff_marginals_fn_jit = []
        self.sensitivity = []

        X = data.to_numpy()
        self.N = X.shape[0]
        self.domain = data.domain

        for cols in self.kway_combinations:

            if self.is_workload_numeric(cols):
                fn, sensitivity = self.get_range_marginal_stats_fn_helper(cols, debug_msg=f'non-jit {cols}: ')
                fn_jit = jax.jit(self.get_range_marginal_stats_fn_helper(cols, debug_msg=f'Compiling {cols}:')[0])

            else:
                fn, sensitivity = self.get_categorical_marginal_stats_fn_helper(cols)
                fn_jit = jax.jit(self.get_categorical_marginal_stats_fn_helper(cols)[0])


            self.true_stats.append(fn(X))
            self.marginals_fn.append(fn)
            self.marginals_fn_jit.append(fn_jit)
            self.sensitivity.append(sensitivity / self.N)

            if not self.IS_REAL_VALUED:
                # Don't add differentiable queries
                diff_fn, sensitivity = self.get_differentiable_stats_fn_helper(cols)
                diff_fn_jit = jax.jit(self.get_differentiable_stats_fn_helper(cols)[0])
                self.diff_marginals_fn.append(diff_fn)
                self.diff_marginals_fn_jit.append(diff_fn_jit)

        get_stats_vmap = lambda X: self.get_stats_jax(X)
        self.get_stats_jax_vmap = jax.vmap(get_stats_vmap, in_axes=(0,))

    def get_num_queries(self):
        return len(self.true_stats)

    def get_dataset_size(self):
        return self.N

    def get_true_stats(self):
        assert self.true_stats is not None, "Error: must call the fit function"
        return jnp.concatenate(self.true_stats)


    def get_stats(self, data: Dataset, indices: list = None):
        X = data.to_numpy()
        stats = []
        I = indices if indices is not None else list(range(self.get_num_queries()))
        for i in I:
            fn = self.marginals_fn[i]
            stats.append(fn(X))
        return jnp.concatenate(stats)

    def get_stats_jit(self, data: Dataset, indices: list = None):
        X = data.to_numpy()
        stats = []
        I = indices if indices is not None else list(range(self.get_num_queries()))
        for i in I:
            fn = self.marginals_fn_jit[i]
            stats.append(fn(X))
        return jnp.concatenate(stats)

    def get_stats_jax(self, X: chex.Array):
        stats = jnp.concatenate([fn(X) for fn in self.marginals_fn])
        return stats

    def get_stats_jax_jit(self, X: chex.Array):
        stats = jnp.concatenate([fn(X) for fn in self.marginals_fn_jit])
        return stats

    def get_diff_stats(self, data: Dataset, indices: list = None):
        assert not self.IS_REAL_VALUED, "Does not work with real-valued data. Must discretize first."
        X = data.to_onehot()
        stats = []
        I = indices if indices is not None else list(range(self.get_num_queries()))
        for i in I:
            fn = self.diff_marginals_fn_jit[i]
            stats.append(fn(X))
        return jnp.concatenate(stats)

    def get_sync_data_errors(self, X):
        assert self.true_stats is not None, "Error: must call the fit function"
        max_errors = []
        for i in range(len(self.true_stats)):
            fn = self.marginals_fn_jit[i]
            error = jnp.abs(self.true_stats[i] - fn(X))
            max_errors.append(error.max())
        return jnp.array(max_errors)

    # def get_sync_data_ave_errors(self, X):
    #     assert self.true_stats is not None, "Error: must call the fit function"
    #     max_errors = []
    #     for i in range(len(self.true_stats)):
    #         fn = self.marginals_fn[i]
    #         error = jnp.linalg.norm(self.true_stats[i] - fn(X), ord=1) / self.true_stats[i].shape[0]
    #         max_errors.append(error)
    #     return jnp.array(max_errors)

    def is_workload_numeric(self, cols):
        for c in cols:
            if c in self.domain.get_numeric_cols():
                return True
        return False

    def get_range_marginal_stats_fn_helper(self, cols, debug_msg=''):
        sizes = []
        for col in cols:
            sizes.append(self.domain.size(col))
        idx = self.domain.get_attribute_indices(cols)
        assert self.is_workload_numeric(cols), "cols tuple must contain at least one numeric attribute"

        cat_idx = self.domain.get_attribute_indices(self.domain.get_categorical_cols())

        bins_sizes = []
        for bins in self.bins:
            sizes_jnp = [jnp.arange(s + 1).astype(float) if i in cat_idx else jnp.linspace(0, 1, bins + 1) for i, s in
                         zip(idx, sizes)]
            bins_sizes.append(sizes_jnp)

        dim = len(self.domain.attrs)
        def stat_fn(X):
            print(debug_msg, X.shape, end='\n')
            X = X.reshape((-1, dim))
            X_proj = X[:, idx]
            stats = []
            for sizes_jnp in bins_sizes:
                stat = jnp.histogramdd(X_proj, sizes_jnp)[0].flatten() / X.shape[0]
                stats.append(stat)
            all_stats = jnp.concatenate(stats)
            return all_stats

        return stat_fn, jnp.sqrt(len(self.bins) * 2)

    def get_categorical_marginal_stats_fn_helper(self, cols):
        """
        Returns marginals function and sensitivity
        :return:
        """
        sizes = []
        for col in cols:
            sizes.append(self.domain.size(col))
        idx = self.domain.get_attribute_indices(cols)
        dim = len(self.domain.attrs)
        assert not self.is_workload_numeric(cols), "Workload cannot contain any numeric attribute"

        sizes_jnp = [jnp.arange(s + 1).astype(float) for i, s in zip(idx, sizes)]
        def stat_fn(x):
            x = x.reshape((-1, dim))
            X_proj = x[:, idx]
            stat = jnp.histogramdd(X_proj, sizes_jnp)[0].flatten()
            return stat

        # def stat_fn(X):


        return stat_fn, jnp.sqrt(2)

    def get_differentiable_stats_fn_helper(self, kway_attributes):
        assert not self.IS_REAL_VALUED, "Does not work with real-valued data. Must discretize first."
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

        return stat_fn, jnp.sqrt(2)
    ##################################################
    ## Adaptive statistics
    ##################################################
    def private_select_measure_statistic(self, key, rho_per_round, X_sync, state: PrivateMarginalsState):
        rho_per_round = rho_per_round / 2

        key, key_em = jax.random.split(key, 2)
        errors = self.get_sync_data_errors(X_sync)
        max_sensitivity = max(self.sensitivity)
        worse_index = exponential_mechanism(key_em, errors, jnp.sqrt(2 * rho_per_round), max_sensitivity)

        key, key_gaussian = jax.random.split(key, 2)
        selected_true_stat = self.true_stats[worse_index]

        sensitivity = self.sensitivity[worse_index]
        sigma_gaussian = float(np.sqrt(sensitivity ** 2 / (2 * rho_per_round)))
        gau_noise = jax.random.normal(key_gaussian, shape=selected_true_stat.shape) * sigma_gaussian

        selected_priv_stat = jnp.clip(selected_true_stat + gau_noise, 0, 1)

        state.add_stats(selected_true_stat, selected_priv_stat, self.marginals_fn[worse_index],
                        self.marginals_fn_jit[worse_index],
                        self.diff_marginals_fn[worse_index] if not self.IS_REAL_VALUED else None,
                        self.diff_marginals_fn_jit[worse_index] if not self.IS_REAL_VALUED else None,
                        )

        return state

    def get_private_statistics(self, key, rho):

        state = PrivateMarginalsState()

        rho_per_stat = rho / len(self.true_stats)
        for i in range(len(self.true_stats)):
            true_stat = self.true_stats[i]
            key, key_gaussian = jax.random.split(key, 2)
            sensitivity = self.sensitivity[i]

            sigma_gaussian = float(np.sqrt(sensitivity ** 2 / (2 * rho_per_stat)))
            gau_noise = jax.random.normal(key_gaussian, shape=true_stat.shape) * sigma_gaussian
            priv_stat = jnp.clip(true_stat + gau_noise, 0, 1)

            state.add_stats(true_stat, priv_stat, self.marginals_fn[i], self.marginals_fn_jit[i],
                            self.diff_marginals_fn[i] if not self.IS_REAL_VALUED else None,
                            self.diff_marginals_fn_jit[i] if not self.IS_REAL_VALUED else None,
                            )

        return state



def exponential_mechanism(key:jnp.ndarray, scores: jnp.ndarray, eps0: float, sensitivity: float):
    dist = jax.nn.softmax(2 * eps0 * scores / (2 * sensitivity))
    cumulative_dist = jnp.cumsum(dist)
    max_query_idx = jnp.searchsorted(cumulative_dist, jax.random.uniform(key, shape=(1,)))
    return max_query_idx[0]


######################################################################
## TEST
######################################################################


def test_discrete():
    num_bins= 3
    import pandas as pd
    cols = ['A', 'B', 'C']
    dom = Domain(cols, [3, 1, 1])

    raw_data_array = pd.DataFrame([
                        [0, 0.0, 0.95],
                        [1, 0.1, 0.90],
                        [2, 0.25, 0.01]], columns=cols)
    # data = Dataset.synthetic_rng(dom, data_size, rng)
    data = Dataset(raw_data_array, dom)
    numeric_features = data.domain.get_numeric_cols()
    data_disc = data.discretize(num_bins=num_bins)
    data_num = Dataset.to_numeric(data=data_disc, numeric_features=numeric_features)

    stat_mod = Marginals.get_all_kway_combinations(data.domain, 3, bins=num_bins)
    stat_mod.fit(data)

    disc_stat_mod = Marginals.get_all_kway_combinations(data_disc.domain, 3, bins=num_bins)
    disc_stat_mod.fit(data_disc)

    stats1 = stat_mod.get_true_stats()
    stats2 = disc_stat_mod.get_true_stats()
    stats_diff = jnp.linalg.norm(stats1 - stats2, ord=1)
    assert stats_diff<=1e-9
    print('test_discrete() passed!')



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

    stat_mod = Marginals.get_all_kway_combinations(domain, 2, bins=[4])
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

    stat_mod = Marginals.get_all_kway_combinations(domain, 2)
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

    print(f'Fitting')
    stime = time.time()
    stat_mod.fit(data)
    etime = time.time() - stime
    print(f'fit elapsed time = {etime:.5f}')

    print(f'num queries = {stat_mod.get_num_queries()}')

    true_stats = stat_mod.get_true_stats()
    print(f'true_stats.shape = ', true_stats.shape)
    print(f'Testing sub stat module evaluation time.')
    print(f'Testing submodule with 5 workloads...')

    sync_data = Dataset.synthetic(domain, 500, 0)

    # stat_fn = jax.jit(sub_stats_moudule.get_stats_fn())
    stime = time.time()
    for it in range(0, 101):
        stat = stat_mod.get_stats(sync_data, indices=[0, 1, 2, 3, 4])
        stat.block_until_ready()
        # print(stat)
        del stat
        if it % 50 == 0:
            print(f'1) it={it:02}. time = {time.time()-stime:.5f}')
    print(f'first evaluate elapsed time = {time.time() - stime:.5f}')

    print(f'Testing submodule with one more workload...')
    stime = time.time()
    for it in range(0, 101):
        D_temp = stat_mod.get_stats(sync_data, indices=[0, 1, 2, 3, 4, 5])
        del D_temp
        if it % 50 == 0:
            print(f'2) it={it:02}. time = {time.time()-stime:.5f}')
    print(f'second evaluate elapsed time = {time.time() - stime:.5f}')

def test_row_answers():

    import numpy as np
    import pandas as pd
    print('debug')
    cols = ['A']
    domain = Domain(cols, [1])

    A = pd.DataFrame(np.array([
        [ 0.4999],
        [ 0.999],
    ]), columns=cols)
    data = Dataset(A, domain=domain)

    stat_mod = Marginals.get_all_kway_combinations(domain, 1, bins=[2])
    stat_mod.fit(data)

    stats1 = stat_mod.get_true_stats()

    row_answers = stat_mod.get_row_answers(data)
    print(stats1.shape)
    print(stats1)
    print(f'row_answers.shape:')
    print(row_answers.shape)
    print(f'row_answers:')
    print(row_answers)

if __name__ == "__main__":
    # test_mixed()
    test_runtime()
    # test_cat_and_diff()
    # test_discrete()
    # test_row_answers()