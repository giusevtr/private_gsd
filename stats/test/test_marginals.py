from unittest import TestCase
from stats.get_marginals_fn import get_marginal_query
from utils import Dataset, Domain
import numpy as np


class TestMarginals(TestCase):

    # @classmethod
    # def setUpClass(cls) -> None:
    #     cls.example_df = pd.read_csv(cls.input_data_path)
    #     cls.aim = AIMSynthesizer()

    def test_fit(self):

        domain = Domain({
            'A':{'type':'categorical', 'size': 10},
            'E':{'type':'ordinal', 'size': 10},
            'F': {'type': 'numerical', 'size': 1},
            'G': {'type': 'numerical', 'size': 1},
        })
        data = Dataset.synthetic(domain, 1000, 0)
        data.df.loc[:, ['F']] = np.zeros((1000, 1))
        data.df.loc[:, ['G']] = np.ones((1000, 1))
        stats0, marginal_fn = get_marginal_query(data, domain, 1, verbose=True, min_bin_density=0.00)
        stats = marginal_fn(data.to_numpy())
        print(stats0)
        print(stats)
        assert np.linalg.norm(stats - stats0) == 0

