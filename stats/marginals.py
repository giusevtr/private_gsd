import itertools
import jax.numpy as jnp
from utils import Dataset, Domain
from stats import Statistic


class Marginals(Statistic):
    def __init__(self, domain, kway_combinations, name='Marginals'):
        super().__init__(domain, name)
        self.kway_combinations = kway_combinations

        # For computing marginal queries
        I = []
        V = []
        for kway_attributes in self.kway_combinations:
            # indices = [domain.get_attribute_onehot_indices(att) for att in kway_attributes]
            col_indices = domain.get_attribute_indices(kway_attributes)
            values = [jnp.arange(0, domain.size(att)) for att in kway_attributes]

            for tup in itertools.product(*values):
                V.append(tup)
                I.append(col_indices)

        self.I = jnp.array(I)
        self.V = jnp.array(V)



        # For computing differentiable marginal queries
        queries = []
        for kway_attributes in self.kway_combinations:
            indices = [domain.get_attribute_onehot_indices(att) for att in kway_attributes]
            for tup in itertools.product(*indices):
                queries.append(tup)
        queries = jnp.array(queries)
        self.queries_split = jnp.array_split(queries, 10)

    def __str__(self):
        return self.name

    def get_sensitivity(self):
        return jnp.sqrt(len(self.kway_combinations))

    @staticmethod
    def get_all_kway_combinations(domain, k):
        cat_columns = domain.get_categorical_cols()
        kway_combinations = list(itertools.combinations(cat_columns, k))
        return Marginals(domain, kway_combinations, name=f'{k}-way Marginals')

    def get_stats_fn(self):
        def stat_fn(X):
            temp1 = X[:, self.I] == self.V
            temp2 = jnp.prod(temp1, axis=2)
            stat = temp2.mean(axis=0)
            return stat

        return stat_fn

    def get_differentiable_stats_fn(self):
        def stat_fn(X):
            return (
                    jnp.concatenate(
                        [jnp.prod(X[:, q], 2).sum(0) for q in self.queries_split]
                    )
                    / X.shape[0]
            )
            # return jnp.einsum("ij,ik,il->jkl", X, X, X) / X.shape[0]
        return stat_fn
        # return self.get_stats_fn()



# def test_discrete_data():
#     DATA_SIZE = 100
#     print('debug')
#     cols = ['A', 'B', 'C', 'D', 'E']
#     domain = Domain(cols, [16, 7, 2, 2, 1])
#     data = Dataset.synthetic(domain, 100, 0)
#     X = data.to_onehot()
#
#     stat_mod = Marginals.get_all_kway_combinations(domain, 2)
#
#     fn1 = stat_mod.get_stats_fn()
#     stats1 = fn1(X)
#     print(stats1.shape)


if __name__ == "__main__":
    test_discrete_data()