import itertools
import jax
import jax.numpy as jnp
from utils import Dataset, Domain
from stats_v2 import Statistic


class Marginals(Statistic):
    true_stats: list = None
    N: int = None  # Dataset size

    def __init__(self, domain, kway_combinations, name='Marginals'):
        super().__init__(domain, name)
        self.kway_combinations = kway_combinations

        self.marginal_functions = []
        for cols in (kway_combinations):
            sizes = []
            for col in cols:
                sizes.append(self.domain.size(col))
            indices = self.domain.get_attribute_indices(cols)
            self.marginal_functions.append(self.get_marginal_stats_fn_helper(indices, sizes))


    @staticmethod
    def get_all_kway_combinations(domain, k):
        cat_columns = domain.get_categorical_cols()
        kway_combinations = [list(idx) for idx in itertools.combinations(cat_columns, k)]
        return Marginals(domain, kway_combinations, name=f'{k}-way Marginals')

    def fit(self, data: Dataset):
        X = data.to_numpy()
        self.N = X.shape[0]
        self.true_stats = [fn(X) for fn in self.marginal_functions]

        # Jit marginals after fitting
        self.marginal_functions = [jax.jit(fn) for fn in self.marginal_functions]

    def get_num_queries(self):
        return len(self.kway_combinations)

    def get_dataset_size(self):
        return self.N

    def get_sub_true_stats(self, index: list):
        assert self.true_stats is not None, "Error: must call the fit function"
        sub_stats = [self.true_stats[i].reshape(-1) for i in index]
        return jnp.concatenate(sub_stats)

    def get_true_stats(self) -> jnp.array:
        assert self.true_stats is not None, "Error: must call the fit function"
        all_indices = [i for i in range(len(self.kway_combinations))]
        return self.get_sub_true_stats(all_indices)

    def get_sync_data_errors(self, X):
        assert self.true_stats is not None, "Error: must call the fit function"
        errors = []
        for i in range(len(self.kway_combinations)):
            fn = self.marginal_functions[i]
            m = self.true_stats[i].shape[0]
            error = jnp.abs(self.true_stats[i] - fn(X)).max()
            errors.append(error)
        return jnp.array(errors)

    def get_sub_stat_module(self, indices: list):
        sub_marginal = Marginals(self.domain, [self.kway_combinations[i] for i in indices], name=self.name)
        if self.true_stats is not None:
            # This case happens if .fit() is called
            # Copy evaluated statistics
            sub_marginal.true_stats = [self.true_stats[i].reshape(-1) for i in indices]
            # Copy jitted functions
            sub_marginal.marginal_functions = [self.marginal_functions[i] for i in indices]
            sub_marginal.N = self.N
        return sub_marginal

    def get_sensitivity(self):
        return jnp.sqrt(len(self.kway_combinations)) / self.N

    def get_marginal_stats_fn_helper(self, idx, sizes):
        # @jax.jit
        def stat_fn(X):
            X_proj = X[:, idx].astype(int)
            stat = jnp.histogramdd(X_proj, sizes)[0].flatten() / X.shape[0]
            return stat

        return stat_fn

    def get_stats_fn(self):

        # @jax.jit
        def stat_fn(X):
            # the functions are Jitted only if .fit() is called
            stats = [fn(X) for fn in self.marginal_functions]
            return jnp.concatenate(stats)
        return stat_fn

    def get_differentiable_stats_fn(self):
        # For computing differentiable marginal queries
        queries = []
        for kway_attributes in self.kway_combinations:
            indices = [self.domain.get_attribute_onehot_indices(att) for att in kway_attributes]
            for tup in itertools.product(*indices):
                queries.append(tup)
        self.queries = jnp.array(queries)
        queries_split = jnp.array_split(self.queries, 10)

        @jax.jit
        def stat_fn(X):
            return jnp.concatenate(
                        [jnp.prod(X[:, q], 2).sum(0) for q in queries_split]
                    ) / X.shape[0]
        return stat_fn




######################################################################
## TEST
######################################################################

def test_discrete_data():
    import numpy as np
    DATA_SIZE = 100
    print('debug')
    cols = ['A', 'B', 'C', 'D', 'E']
    domain = Domain(cols, [16, 7, 2, 2, 1])
    data = Dataset.synthetic(domain, DATA_SIZE, 0)
    X = data.to_numpy()
    X_oh = data.to_onehot()

    stat_mod = Marginals.get_all_kway_combinations(domain, 2)

    fn1 = stat_mod.get_stats_fn()
    fn2 = stat_mod.get_differentiable_stats_fn()
    stats1 = fn1(X)
    stats2 = fn2(X_oh)
    # print(stats1.shape)

    diff = jnp.abs(stats2-stats1).max()
    assert diff<0.00001, 'differentialble stats must match'
    print('TEST PASSED')

def test_runtime():
    import time
    DATA_SIZE = 10000
    cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    domain = Domain(cols, [16, 7, 10, 15, 10, 13,  1, 12, 8, 10, 6])
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

    sub_stats_moudule = stat_mod.get_sub_stat_module(indices=[0, 1, 2, 3, 4])
    sync_X = sync_data.to_numpy()
    # stat_fn = jax.jit(sub_stats_moudule.get_stats_fn())
    stat_fn = sub_stats_moudule.get_stats_fn()
    stime = time.time()
    for it in range(0, 101):
        D_temp = stat_fn(sync_X)
        del D_temp
        if it % 50 == 0:
            print(f'1) it={it:02}. time = {time.time()-stime:.5f}')
    print(f'first evaluate elapsed time = {time.time() - stime:.5f}')

    print(f'Testing submodule with one more workload...')
    sub_stats_moudule_2 = stat_mod.get_sub_stat_module(indices=[0, 1, 2, 3, 4, 5])
    stat_fn_2 = sub_stats_moudule_2.get_stats_fn()
    stime = time.time()
    for it in range(0, 101):
        D_temp = stat_fn_2(sync_X)
        del D_temp
        if it % 50 == 0:
            print(f'2) it={it:02}. time = {time.time()-stime:.5f}')
    print(f'second evaluate elapsed time = {time.time() - stime:.5f}')




if __name__ == "__main__":
    test_discrete_data()
    # test_runtime()