import itertools
import jax
import jax.numpy as jnp
from utils import Dataset, Domain
from utils.utils_data import Domain
from stats.private_statistics import PrivateMarginalsState
import numpy as np
import chex


# class PrivateMarginalsState:
#     def __init__(self):
#         self.NUM_STATS = 0
#         self.true_stats = []
#         self.priv_stats = []
#         self.priv_marginals_fn = []
#         self.priv_diff_marginals_fn = []
#         self.selected_marginals = []
#
#         self.priv_loss_l2_fn_jit_list = []
#         self.priv_loss_l2_fn_jit_vmap_list = []
#
#     def add_stats(self, true_stat, priv_stat, marginal_fn, diff_marginal_fn):
#         self.NUM_STATS += true_stat.shape[0]
#         self.true_stats.append(true_stat)
#         self.priv_stats.append(priv_stat)
#         self.priv_marginals_fn.append(marginal_fn)
#         self.priv_diff_marginals_fn.append(diff_marginal_fn)
#
#         # priv_loss_fn = lambda X: jnp.linalg.norm(priv_stat - marginal_fn(X), ord=2) / priv_stat.shape[0]
#         priv_loss_fn = lambda X: jnp.linalg.norm(priv_stat - marginal_fn(X), ord=2)
#         priv_loss_fn_jit = jax.jit(priv_loss_fn)
#         self.priv_loss_l2_fn_jit_list.append(priv_loss_fn_jit)
#
#         priv_loss_fn_jit_vmap = jax.vmap(priv_loss_fn_jit, in_axes=(0, ))
#         self.priv_loss_l2_fn_jit_vmap_list.append(jax.jit(priv_loss_fn_jit_vmap))
#         # fitness_fn_vmap = lambda x_pop: compute_error_vmap(x_pop)
#         # fitness_fn_vmap_jit = jax.jit(fitness_fn)
#
#     def get_true_stats(self):
#         return jnp.concatenate(self.true_stats)
#
#     def get_priv_stats(self):
#         return jnp.concatenate(self.priv_stats)
#
#     def get_stats(self, X):
#         return jnp.concatenate([fn(X) for fn in self.priv_marginals_fn])
#
#     def get_diff_stats(self, X):
#         return jnp.concatenate([diff_fn(X) for diff_fn in self.priv_diff_marginals_fn])
#
#     def true_loss_inf(self, X):
#         true_stats_concat = jnp.concatenate(self.true_stats)
#         sync_stats_concat = self.get_stats(X)
#         return jnp.abs(true_stats_concat - sync_stats_concat).max()
#
#     def true_loss_l2(self, X):
#         true_stats_concat = jnp.concatenate(self.true_stats)
#         # loss = jnp.sum(jnp.array([fn_jit(X)  for fn_jit in self.priv_loss_l2_fn_jit]))
#         sync_stats_concat = self.get_stats(X)
#         return jnp.linalg.norm(true_stats_concat - sync_stats_concat, ord=2) / true_stats_concat.shape[0]
#
#     def priv_loss_inf(self, X):
#         priv_stats_concat = jnp.concatenate(self.priv_stats)
#         sync_stats_concat = self.get_stats(X)
#         return jnp.abs(priv_stats_concat - sync_stats_concat).max()
#
#     def priv_loss_l2(self, X):
#         priv_stats_concat = jnp.concatenate(self.priv_stats)
#         sync_stats_concat = self.get_stats(X)
#         return jnp.linalg.norm(priv_stats_concat - sync_stats_concat, ord=2) / priv_stats_concat.shape[0]
#
#     def priv_loss_l2_jit(self, X):
#         loss = 0
#         for jit_fn in self.priv_loss_l2_fn_jit_list:
#             loss += jit_fn(X)
#         return loss / self.NUM_STATS
#
#
#     def priv_marginal_loss_l2_jit(self, X):
#         losses = []
#         # for jit_fn in self.priv_loss_l2_fn_jit_list:
#         for i in range(len(self.priv_loss_l2_fn_jit_list)):
#             stat_size = self.priv_stats[i].shape[0]
#             jit_fn = self.priv_loss_l2_fn_jit_list[i]
#             losses.append(jit_fn(X)/stat_size)
#         return losses
#
#     def priv_loss_l2_vmap_jit(self, X_pop):
#         loss = None
#         for jit_fn in self.priv_loss_l2_fn_jit_vmap_list:
#             this_loss = jit_fn(X_pop)
#             loss = this_loss if loss is None else loss + this_loss
#         return loss / self.NUM_STATS if loss is not None else 0
#
#
#     def priv_diff_loss_inf(self, X_oh):
#         priv_stats_concat = jnp.concatenate(self.priv_stats)
#         sync_stats_concat = self.get_diff_stats(X_oh)
#         return jnp.abs(priv_stats_concat - sync_stats_concat).max()
#
#     def priv_diff_loss_l2(self, X_oh):
#         priv_stats_concat = jnp.concatenate(self.priv_stats)
#         sync_stats_concat = self.get_diff_stats(X_oh)
#         return jnp.linalg.norm(priv_stats_concat - sync_stats_concat, ord=2) ** 2 / priv_stats_concat.shape[0]



class Marginals:

    def __init__(self, domain, kway_combinations, bins=(32,)):
        self.domain = domain
        self.kway_combinations = kway_combinations
        self.bins = list(bins)

        # Check that domain constains real-valued features
        self.IS_REAL_VALUED = len(domain.get_numeric_cols()) > 0

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
        self.row_answer_fn = []
        self.diff_marginals_fn = []
        self.sensitivity = []

        X = data.to_numpy()
        self.N = X.shape[0]
        self.domain = data.domain

        for cols in self.kway_combinations:
            sizes = []
            for col in cols:
                sizes.append(self.domain.size(col))
            indices = self.domain.get_attribute_indices(cols)
            fn, sensitivity = self.get_marginal_stats_fn_helper(indices, sizes)
            row_answer_fn, sensitivity = self.get_marginal_stats_fn_helper(indices, sizes)
            # row_answer_fn_vmap = jax.vmap(row_answer_fn, in_axes=(0, ), out_axes=(0, ))
            row_answer_fn_vmap = jax.vmap(row_answer_fn, in_axes=(0, ))
            self.true_stats.append(fn(X))
            self.marginals_fn.append(jax.jit(fn))

            self.row_answer_fn.append(jax.jit(row_answer_fn_vmap))

            self.sensitivity.append(sensitivity / self.N)

            if not self.IS_REAL_VALUED:
                # Don't add differentiable queries
                diff_fn, sensitivity = self.get_differentiable_stats_fn_helper(cols)
                self.diff_marginals_fn.append(diff_fn)

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


    def get_stats_jax(self, X: chex.Array):
        stats = jnp.concatenate([fn(X) for fn in self.marginals_fn])
        return stats

    def get_row_answers(self, data: Dataset, indices: list = None):
        X = data.to_numpy()
        stats = []
        I = indices if indices is not None else list(range(self.get_num_queries()))
        for i in I:
            fn = self.row_answer_fn[i]
            stats.append(fn(X))
        return jnp.concatenate(stats, axis=1)

    def get_diff_stats(self, data: Dataset, indices: list = None):
        assert not self.IS_REAL_VALUED, "Does not work with real-valued data. Must discretize first."
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
        for i in range(len(self.true_stats)):
            fn = self.marginals_fn[i]
            error = jnp.abs(self.true_stats[i] - fn(X))
            max_errors.append(error.max())
        return jnp.array(max_errors)

    def get_sync_data_ave_errors(self, X):
        assert self.true_stats is not None, "Error: must call the fit function"
        max_errors = []
        for i in range(len(self.true_stats)):
            fn = self.marginals_fn[i]
            error = jnp.linalg.norm(self.true_stats[i] - fn(X), ord=1) / self.true_stats[i].shape[0]
            max_errors.append(error)
        return jnp.array(max_errors)

    # def get_sensitivity(self):
    #     return jnp.sqrt(2) / self.N

    def get_marginal_stats_fn_helper(self, idx, sizes) :
        """
        Returns marginals function and sensitivity
        :return:
        """
        is_numeric = False
        dim = len(self.domain.attrs)
        cat_idx = self.domain.get_attribute_indices(self.domain.get_categorical_cols())
        for i in idx:
            if i not in cat_idx:
                is_numeric = True

        if is_numeric:
            bins_sizes = []
            for bins in self.bins:
                sizes_jnp = [jnp.arange(s + 1).astype(float) if i in cat_idx else jnp.linspace(0, 1, bins+1) for i, s in zip(idx, sizes)]
                bins_sizes.append(sizes_jnp)

            def stat_fn(X):

                X = X.reshape((-1, dim))
                X_proj = X[:, idx]
                stats = []
                for sizes_jnp in bins_sizes:
                    # stat_fn_list.append(get_stat_fn(idx, sizes_jnp))
                    stat = jnp.histogramdd(X_proj, sizes_jnp)[0].flatten() / X.shape[0]
                    stats.append(stat)
                all_stats = jnp.concatenate(stats)
                return all_stats

            return stat_fn, jnp.sqrt(len(self.bins) *2)

        else:
            sizes_jnp = [jnp.arange(s + 1).astype(float) for i, s in zip(idx, sizes)]
            def stat_fn(X):
                # X_proj = (X[:, idx]).astype(int)
                X = X.reshape((-1, dim))
                X_proj = X[:, idx]
                stat = jnp.histogramdd(X_proj, sizes_jnp)[0].flatten() / X.shape[0]
                return stat

            return stat_fn, jnp.sqrt(2)
        #
        # return stat_fn_list

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
                        self.row_answer_fn[worse_index],
                        self.diff_marginals_fn[worse_index] if not self.IS_REAL_VALUED else None)

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

            state.add_stats(true_stat, priv_stat, self.marginals_fn[i], self.row_answer_fn[i],
                            self.diff_marginals_fn[i] if not self.IS_REAL_VALUED else None)

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

def test_real_and_diff():

    import numpy as np
    import pandas as pd
    print('debug')
    cols = ['A', 'B']
    domain = Domain(cols, [1, 1])

    A = pd.DataFrame(np.array([
        [0.0, 1.0],
        [0.1, 0.1],
        [0.1, 0.2],
        [0.0, 0.3],
    ]), columns=cols)
    data = Dataset(A, domain=domain)
    X = data.to_numpy()

    stat_mod = Marginals.get_all_kway_combinations(domain, 2, bins=3)
    stat_mod.fit(data)

    stats_true = stat_mod.get_true_stats()
    stats_diff = stat_mod.get_diff_stats(data)

    print('true_stats - diff_stats', jnp.linalg.norm(stats_true - stats_diff, ord=1))
    print(stats_true)
    print(stats_diff)


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
    # test_runtime()
    # test_real_and_diff()
    # test_discrete()
    test_row_answers()