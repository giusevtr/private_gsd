import jax.random

from utils import Dataset, Domain
from stats import Marginals, NullCounts
import pandas as pd
import json

def test_dataset():
    config = json.load(open('../tests/domain.json'))
    domain = Domain(config)

    data = Dataset.synthetic_jax_rng(domain, N=5, rng=jax.random.PRNGKey(0), null_values=0.5)
    print(data)


def test_marginals():
    # test_onehot_encoding()
    # test_discrete()

    df = pd.read_csv('../tests/data.csv')
    config = json.load(open('../tests/domain.json'))

    print(df)
    print(config)
    domain = Domain(config)

    data = Dataset(df, domain)

    print(data.to_numpy())

    # module = Marginals.get_all_kway_combinations(domain, k=1, bins=[2, 4])
    module = Marginals(domain, kway_combinations=[('B', )], k=1, bins=[2, 4])
    stat_fn = module._get_dataset_statistics_fn()


    stats = stat_fn(data)

    print('Attribute B 1-way stats: ', stats)

    null_counter = NullCounts(domain)
    null_fn = null_counter._get_dataset_statistics_fn()
    null_counts = null_fn(data)
    print('null_counts: ', null_counts)


if __name__ == "__main__":
    # test_marginals()
    test_dataset()