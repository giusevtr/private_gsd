from unittest import TestCase

import jax
import pandas as pd
import jax.numpy as jnp
from stats.get_marginals_fn_v2 import get_thresholds_realvalued, _get_query_params, \
    _get_thresholds_1way_marginal_fn, _get_stats_fn, _get_mixed_marginal_fn, get_thresholds_categorical
from utils import Dataset, Domain
import numpy as np


class TestMarginals(TestCase):

    @classmethod
    def setUpClass(self) -> None:
        self.domain = Domain({
            'A':{'type':'categorical', 'size': 2},
            'B': {'type': 'numerical', 'size': 1},
            'C': {'type': 'numerical', 'size': 1},
        })
        self.data = Dataset.synthetic(self.domain, 10000, 0)


    def test_thresholds(self):
        data_df = pd.Series([0.2, 0.2, 0.4, 0.45, 0, 0, 0, 0, 0, 0.7])
        bins = get_thresholds_realvalued(data_df, 0.1)
        print(bins)

    def test_query_params(self):
        domain = Domain({
            'A':{'type':'categorical', 'size': 2},
            'B': {'type': 'numerical', 'size': 1},
            'C': {'type': 'numerical', 'size': 1},
        })
        data = Dataset.synthetic(domain, 1000, 0)

        idx = [1]
        bins = [np.array([0, 0.1, 0.2, 0.4, 1])]
        a, params_1way = _get_query_params(data, idx, bins)
        stat_fn = _get_stats_fn(k=1, query_params=params_1way)
        assert np.abs(np.array(a) - np.array(stat_fn(data.to_numpy())))

        idx = [1, 2]
        bins = 2*[np.array([0, 0.1, 0.2, 0.4, 1])]
        a, params_2way = _get_query_params(data, idx, bins)
        stat_fn = _get_stats_fn(k=1, query_params=params_1way)
        assert np.abs(np.array(a) - np.array(stat_fn(data.to_numpy())))

    def test_categorical_thresholds(self):
        feature = 'A'
        bins = get_thresholds_categorical(self.data.df[feature], self.domain.size(feature))
        X = self.data.to_numpy()
        indices = [self.domain.get_attribute_index(feature)]
        ans, q_params = _get_query_params(self.data, indices, [bins[0]['bins']])
        query_fn = _get_stats_fn(k=1, query_params=q_params)
        ans2 = query_fn(X)
        assert np.abs(np.array(ans) - np.array(ans2)).max() < 1e-7


    def test_query_params_and_thresholds(self):
        feature = 'B'
        bins = get_thresholds_realvalued(self.data.df[feature], 0.001, levels=20)
        indices = [self.domain.get_attribute_index(feature)]
        X = self.data.to_numpy()
        for level in range(bins['levels']):
            bin_edges = [bins[level]['bins']]
            priv_stats = bins[level]['stats']
            ans, q_params = _get_query_params(self.data, indices, bin_edges)
            query_fn = _get_stats_fn(k=1, query_params=q_params)
            ans2 = query_fn(X)
            assert np.abs(np.array(priv_stats)-np.array(ans)).max()<1e-9
            assert np.abs(np.array(priv_stats)-ans2).max()<1e-9

    def test_query_params_and_thresholds_2(self):
        domain = Domain({
            'A': {'type': 'numerical', 'size': 1},
            'B': {'type': 'numerical', 'size': 1},
            'C': {'type': 'numerical', 'size': 1},
            'D': {'type': 'ordinal', 'size': 200},
        })
        N = 10
        arr = np.column_stack((np.zeros((N, 1)), 0.5*np.ones((N, 1)), np.ones((N, 1))))
        ord = np.random.randint(low=0, high=200, size=(N, 1))
        df = pd.DataFrame(np.column_stack((arr, ord)), columns=['A', 'B', 'C', 'D'])
        data = Dataset(df, domain)

        for feature in ['A', 'B', 'C']:
            bins = get_thresholds_realvalued(data.df[feature], 0.00, levels=1)
            indices = [domain.get_attribute_index(feature)]
            X = data.to_numpy()
            for level in range(bins['levels']):
                bin_edges = [bins[level]['bins']]
                stats = np.array(bins[level]['stats'])
                stats2, q_params = _get_query_params(data, indices, bin_edges)
                query_fn = _get_stats_fn(k=1, query_params=q_params)
                stats_query_fn = np.array(query_fn(X))
                error1 = np.abs(stats - np.array(stats2))
                error2 = np.abs(stats - stats_query_fn)
                print(f'Feature {feature}:')
                print(f'stats       = ', stats)
                print(f'stats2      = ', stats2)
                print(f'query_fn(D) = ', stats_query_fn)
                print('|priv_stats - ans|  =', error1)
                print('|priv_stats - ans2| = ', error2)
                assert  error1.max()< 1e-9
                assert error2.max() < 1e-9
                print()
    def test_1_way_marginals(self):

        stats_1_way, stat_1_way_fn, bins = _get_thresholds_1way_marginal_fn(self.data)
        assert (jnp.array(stats_1_way) - stat_1_way_fn(self.data.to_numpy())).max() < 1e-6

    def test_realvalued_k_marginals(self):

        print('Test 1')
        stats_1_way, stat_1_way_fn, bins = _get_thresholds_1way_marginal_fn(self.data)
        p_stats, stat_fn = _get_mixed_marginal_fn(self.data, k_real=2, bin_edges=bins, verbose=True)
        assert (jnp.array(p_stats) - stat_fn(self.data.to_numpy())).max() < 1e-6

        # Test max_size
        print('Test with max size=100')
        p_stats, stat_fn = _get_mixed_marginal_fn(self.data, k_real=2, bin_edges=bins, maximum_size=1000, verbose=True)
        assert (jnp.array(p_stats) - stat_fn(self.data.to_numpy())).max() < 1e-6

        # with conditional columns
        p_stats, stat_fn = _get_mixed_marginal_fn(self.data, k_real=2, bin_edges=bins, maximum_size=1000, conditional_column=['A'], verbose=True)
        assert (jnp.array(p_stats) - stat_fn(self.data.to_numpy())).max() < 1e-6


        # Test privacy

    def test_realvalued_k_marginals_zcdp(self):

        stats_1_way, stat_1_way_fn, bins = _get_thresholds_1way_marginal_fn(self.data, rho=10, verbose=True)
        p_stats, stat_fn = _get_mixed_marginal_fn(self.data, k_real=2, bin_edges=bins,
                                                  maximum_size=1000, conditional_column=['A'], rho=0.5, verbose=True)
        error = (jnp.array(p_stats) - stat_fn(self.data.to_numpy()))
        print(error.mean(), error.max())
        assert error.max() > 1e-6

    def test_low_entropy_data(self):
        domain = Domain({
            'A': {'type': 'numerical', 'size': 1},
            'B': {'type': 'numerical', 'size': 1},
        })
        rng = np.random.default_rng(0)
        N = 10000
        arr = 0.5 * np.ones((10000, 2)) + rng.normal(0, 0.001, size=(N, 2))
        df = pd.DataFrame(arr, columns=['A', 'B'])
        data = Dataset(df, domain)
        stats_1_way, stat_1_way_fn, bins = _get_thresholds_1way_marginal_fn(data, verbose=True)

        counts1 = stats_1_way * N
        counts2 = stat_1_way_fn(data.to_numpy()) * N
        error = jnp.abs(counts1 - counts2)
        assert error.max() < 1e-1

        p_stats, stat_fn = _get_mixed_marginal_fn(data, k_real=2, bin_edges=bins,
                                                  maximum_size=1000,
                                                  verbose=True)
        counts1 = np.array(p_stats) * N
        counts2 = np.array(stat_fn(data.to_numpy())) * N
        error = np.abs(counts1 - counts2)
        assert error.max() < 1e-1

    def test_1_way_marginals2(self):
        domain = Domain({'A': {'type': 'numerical', 'size': 1},})
        N = 10
        arr = np.concatenate((np.zeros(N), 0.5 * np.ones(N), np.ones(N)))
        df = pd.DataFrame(arr, columns=['A'])
        data = Dataset(df, domain)

        stats_1_way, stat_1_way_fn, bins = _get_thresholds_1way_marginal_fn(data, levels=1, verbose=True)


        stats_debug1 = np.array(stats_1_way)
        stats_debug2 = np.array(stat_1_way_fn(data.to_numpy()))

        error = (stats_debug1 - stats_debug2)
        print(stats_debug1)
        print(stats_debug2)
        print(error.mean(), error.max())
        assert error.max() < 1e-8



    def test_mixed_marginals(self):
        domain = Domain({
            'A': {'type': 'numerical', 'size': 1},
            'B': {'type': 'numerical', 'size': 1},
            'C': {'type': 'numerical', 'size': 1},
            'D': {'type': 'ordinal', 'size': 20},
            'E': {'type': 'categorical', 'size': 3},
            'Label': {'type': 'categorical', 'size': 2},
        })
        N = 1000
        arr_A = 0.49 * np.ones(N)
        arr_B = np.zeros(N)
        arr_C = np.random.rand(N)
        arr_D = np.random.randint(low=0, high=20, size=N)
        arr_E = np.random.randint(low=0, high=3, size=N)
        # arr_D = np.array([0, 2, 0, 0, 0, 0, 0, 0, 0, 1])
        arr_label = np.random.randint(low=0, high=2, size=N)
        df = pd.DataFrame(np.column_stack((arr_A, arr_B, arr_C, arr_D, arr_E, arr_label)), columns=['A', 'B', 'C', 'D', 'E', 'Label'])
        data = Dataset(df, domain)

        stats_1_way, stat_1_way_fn, bins = _get_thresholds_1way_marginal_fn(data, verbose=True)
        p_stats, stat_fn = _get_mixed_marginal_fn(data, k_real=2, bin_edges=bins,
                                                  maximum_size=2000, conditional_column=[], verbose=True)
        error = (jnp.array(p_stats) - stat_fn(data.to_numpy()))
        print(error.mean(), error.max())
        assert error.max() < 1e-6
