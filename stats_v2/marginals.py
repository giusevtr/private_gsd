import itertools
import jax
import jax.numpy as jnp
from utils import Dataset, Domain
from stats_v2 import Statistic


class Marginals(Statistic):
    true_stats: list

    def __init__(self, domain, kway_combinations, name='Marginals'):
        super().__init__(domain, name)
        self.kway_combinations = kway_combinations

        # For computing marginal queries
        # stime = time.time()
        I = []
        V = []
        self.marginal_functions = []
        for kway_attributes in self.kway_combinations:
            # indices = [domain.get_attribute_onehot_indices(att) for att in kway_attributes]
            col_indices = self.domain.get_attribute_indices(kway_attributes)
            values = [jnp.arange(0, self.domain.size(att)) for att in kway_attributes]

            temp_I = []
            temp_V = []
            for tup in itertools.product(*values):
                temp_V.append(tup)
                temp_I.append(col_indices)
                V.append(tup)
                I.append(col_indices)

            temp_I = jnp.array(temp_I)
            temp_V = jnp.array(temp_V)
            self.marginal_functions.append(self.get_marginal_stats_fn_helper(temp_I, temp_V))

        self.I = jnp.array(I)
        self.V = jnp.array(V)



        # For computing differentiable marginal queries
        queries = []
        for kway_attributes in self.kway_combinations:
            indices = [self.domain.get_attribute_onehot_indices(att) for att in kway_attributes]
            for tup in itertools.product(*indices):
                queries.append(tup)
        self.queries = jnp.array(queries)

    def fit(self, data: Dataset):
        X = data.to_numpy()
        self.true_stats = [fn(X) for fn in self.marginal_functions]

    def get_num_queries(self):
        return len(self.kway_combinations)

    def get_sub_answers(self, index: list):
        assert self.true_stats is not None, "Error: must call the fit function"
        sub_stats = [self.true_stats[i].reshape(-1) for i in index]
        return jnp.concatenate(sub_stats)

    def get_true_stats(self) -> jnp.array:
        assert self.true_stats is not None, "Error: must call the fit function"
        all_indices = [i for i in range(len(self.kway_combinations))]
        return self.get_sub_answers(all_indices)

    def get_sync_data_errors(self, X):
        assert self.true_stats is not None, "Error: must call the fit function"
        errors = []
        for i in range(len(self.kway_combinations)):
            fn = self.marginal_functions[i]
            m = self.true_stats[i].shape[0]
            error = jnp.linalg.norm( jnp.abs(self.true_stats[i] - fn(X)), ord=1)  / m
            errors.append(error)
        return jnp.array(errors)

    def get_sub_stat_module(self, indices: list):
        sub_marginal = Marginals(self.domain, [self.kway_combinations[i] for i in indices])
        if self.true_stats is not None:
            sub_marginal.true_stats = [self.true_stats[i].reshape(-1) for i in indices]
        return sub_marginal


    def get_sensitivity(self):
        return jnp.sqrt(len(self.kway_combinations))

    @staticmethod
    def get_all_kway_combinations(domain, k):
        cat_columns = domain.get_categorical_cols()
        kway_combinations = list(itertools.combinations(cat_columns, k))
        return Marginals(domain, kway_combinations, name=f'{k}-way Marginals')

    def get_marginal_stats_fn_helper(self, I, V):
        @jax.jit
        def stat_fn(X):
            # return jnp.array([jnp.prod(X[:, self.I] == self.V, axis=2).sum(0)] ) / X.shape[0]
            return jnp.prod(X[:, I] == V, axis=2).sum(0) / X.shape[0]

        return stat_fn

    def get_stats_fn(self):
        I_split = jnp.array_split(self.I, 10)
        V_split = jnp.array_split(self.V, 10)

        @jax.jit
        def stat_fn(X):
            # return jnp.array([jnp.prod(X[:, self.I] == self.V, axis=2).sum(0)] ) / X.shape[0]
            return jnp.concatenate(
                    [jnp.prod(X[:, i] == v, axis=2).sum(0) for i, v in zip(I_split, V_split)]
                ) / X.shape[0]
        return stat_fn

    def get_differentiable_stats_fn(self):

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
    stime = time.time()
    stat_mod.fit(data)
    etime = time.time() - stime
    print(f'fit elapsed time = {etime:.5f}')


    sync_data = Dataset.synthetic(domain, 500, 0)
    sync_X = sync_data.to_numpy()
    stat_fn = stat_mod.get_stats_fn()
    stime = time.time()
    for _ in range(100):
        D_temp = stat_fn(sync_X)
        del D_temp
    print(f'evaluate elapsed time = {time.time() - stime:.5f}')

if __name__ == "__main__":
    # test_discrete_data()
    test_runtime()