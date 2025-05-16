import pandas as pd
import jax
import itertools
import jax.numpy as jnp
from jax.lib import xla_bridge
import numpy as np
if xla_bridge.get_backend().platform == 'cpu':
    print("For Genetic-SD support, please install jax:pip install -U \"jax[cuda12]\"")
from snsynth.base import Synthesizer
from snsynth.utils import cdp_rho
from snsynth.transform import *
from snsynth.transform.type_map import TypeMap, ColumnTransformer
from genetic_sd.utils.ordinal_transformer import OrdinalTransformer
from genetic_sd.utils import Domain, Dataset
from genetic_sd.adaptive_statistics import AdaptiveChainedStatistics, Marginals
from genetic_sd.generator.generator_genetic_sd import GeneticSD
from genetic_sd.generator.mutation_strategies import AVAILABLE_GENETIC_OPERATORS
from genetic_sd.utils.private_thresholds import get_thresholds_zcdp_adaptive


class GSDSynthesizer(Synthesizer):
    """
    Based on the paper: https://arxiv.org/abs/2306.03257
    """

    def __init__(self, epsilon=1., delta=1e-9, verbose=False, *args, **kwargs):
        # GSDSynthesizer.__init__()
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.verbose = verbose

        self.rho = cdp_rho(epsilon=epsilon, delta=delta)

        self.data = None
        self.sync_data = None
        self.sync_data_df = None
        self.statistic_fn = None

    def fit(
            self,
            data: pd.DataFrame, *ignore,
            meta_data: dict = None,
            transformer=None,
            categorical_columns: list = None,
            ordinal_columns: list = None,
            continuous_columns: list = None,
            preprocessor_eps: float = 0.0,
            nullable=False,
            N_prime: int = None,
            early_stop_threshold: float = 0.0001,  # Increase this if you need to make GSD end sooner
            conditional_columns: list = (),  # Add extra dimension to the marginals. (Must be categorical)
            genetic_operators=(),
            tree_query_depth: int = 3,
            num_discretization_intervals: int = 64,
            continuous_data_granularity: float = 0.001,
            seed=None):

        """
        This function computes the 1st and 2nd moment statistics.
        """

        # ---------------#
        # Preprocess data
        # ---------------#
        self.data = self._get_data(data, meta_data, transformer, categorical_columns, ordinal_columns,
                                   continuous_columns, preprocessor_eps, nullable)
        self.dim = len(self.data.domain.attrs)
        self.N_prime = len(self.data.df) if N_prime is None else N_prime
        self.conditional_columns = conditional_columns

        domain = self.data.domain

        if seed is None: seed = np.random.randint(0, 1e9)
        self.key = jax.random.PRNGKey(seed)



        # Choose columns to discretize
        numeric_columns_discretize = domain.get_continuous_cols()
        for ord_col in domain.get_ordinal_cols():
            if domain.size(ord_col) > num_discretization_intervals:
                numeric_columns_discretize.append(ord_col)
        if self.verbose:
            print(f'Number of discretization columns: {len(numeric_columns_discretize)}')
        # ---------------- #
        #  Compute the number of continuous or high-cardinality ordinal columns
        # and split privacy budget proportionally.
        # ---------------- #
        num_workload_k_1 = len(numeric_columns_discretize)
        num_workload_k_2 = len([list(idx) for idx in itertools.combinations(self.data.domain.attrs, 2)])


        rho_second_moments = self.rho
        num_cols_thresholds = {}
        if num_workload_k_1 > 0:
            rho_1 = self.rho * num_workload_k_1 / (num_workload_k_1 + num_workload_k_2)
            rho_second_moments = self.rho - rho_1
            if self.verbose:
                print(f'Budget:\nBudget for finding numeric columns thresholds: {rho_1:.4}\n'
                        f'Budget for matching second moment statistics: {rho_second_moments:.4}')

            # ---------------#
            # Get numeric column thresholds.
            # For each numeric column, find the set of K intervals that equality divide the data under DP.
            # For example, if K=10, then each interval should cover 10% of the data points.
            # ---------------#
            rho_per_col = rho_1 / len(numeric_columns_discretize)  # privacy budget for each numeric column
            for num_col in numeric_columns_discretize:
                min_val, max_val = domain.range(num_col)
                data_granularity = 1 if domain.is_ordinal(num_col) else continuous_data_granularity

                priv_thres = get_thresholds_zcdp_adaptive(self.data.df[num_col],
                                                          rho=rho_per_col,
                                                          data_range=(min_val, max_val),
                                                          data_granularity=data_granularity,
                                                          num_intervals=num_discretization_intervals,
                                                          verbose=False)
                num_cols_thresholds[num_col] = priv_thres

                # Compute error.
                if self.verbose:
                    print(f'Setting up bin edges for column {num_col}.')
                    values = self.data.df[num_col].values
                    stats = np.histogramdd(values, [priv_thres])[0].flatten() / values.shape[0]
                    max_interval_coverage = stats.max()
                    print(f'Max interval coverage is {max_interval_coverage}. Should be lower than {1/num_discretization_intervals}')


        # ---------------#
        # Initialize genetic generator and statistics
        # ---------------#

        # Define genetic strategy and parameters
        if len(genetic_operators) == 0:
            genetic_operators = AVAILABLE_GENETIC_OPERATORS

        # Set thresholds
        domain.set_bin_edges(num_cols_thresholds)

        # Genetic-SD algorithm
        genetic_sd_generator = GeneticSD(domain=domain,
                                         data_size=self.N_prime,
                                         num_generations=50000000,
                                         genetic_operators=genetic_operators,
                                         print_progress=self.verbose,
                                         stop_early=True,
                                         stop_eary_threshold=early_stop_threshold,
                                         sparse_statistics=True)
        # genetic_sd_generator.fit
        # This object computes the data's statistical properties under differential privacy
        private_stats = AdaptiveChainedStatistics(self.data)

        # Defined 2nd moment statistics based column quantiles computed in the previous step
        private_stats.add_stat_module_and_fit(
            Marginals.get_all_kway_combinations(self.data.domain,
                                                k=2,
                                                tree_query_depth=tree_query_depth,
                                                num_cols_thresholds=num_cols_thresholds))

        # Add noise to 2nd moment statistics using privacy budget rho_2
        self.key, key_dp_2 = jax.random.split(self.key)
        private_stats.private_measure_all_statistics(key=key_dp_2, rho=rho_second_moments)

        self.statistic_fn = private_stats.get_dataset_statistics_fn()
        # Generate synthetic data that matches noisy 2nd moment statistics
        self.key, key_fit_2 = jax.random.split(self.key)
        self.sync_data = genetic_sd_generator.fit(key_fit_2, private_stats)

        # Get errors
        true_stats = private_stats.get_all_true_statistics()
        stat_fn = private_stats.get_dataset_statistics_fn()
        sync_stats = stat_fn(self.sync_data)
        l1_error = jnp.abs(true_stats - sync_stats).mean()
        max_error = jnp.abs(true_stats - sync_stats).max()
        print('Final error', l1_error, max_error)

        data_list = self.get_values_as_list(self.data.domain, self.sync_data.df)
        self.sync_data_df = self._transformer.inverse_transform(data_list)


    def stat_fn(self, data: Dataset):
        return self.statistic_fn(data)

    def get_values_as_list(self, domain: Domain, df: pd.DataFrame):
        data_as_list = []
        for i, row in df.iterrows():
            row_list = []
            for j, col in enumerate(domain.attrs):
                value = row[col]
                if col in domain.get_categorical_cols():
                    row_list.append(int(value))
                if col in domain.get_ordinal_cols():
                    row_list.append(int(value))
                if col in domain.get_continuous_cols():
                    row_list.append(float(value))
            data_as_list.append(row_list)
        return data_as_list

    def sample(self, samples=None):
        if samples is None:
            return self.sync_data_df

        data = self.sync_data_df.sample(n=samples, replace=(samples > self.N_prime))
        return data

    @staticmethod
    def get_column_names(data):
        if isinstance(data, pd.DataFrame):
            return data.columns
        elif isinstance(data, np.ndarray):
            return list(range(len(data[0])))
        elif isinstance(data, list):
            return list(range(len(data[0])))

    def _get_data(self, data: pd.DataFrame,
                  meta_data: dict = None,
                  transformer=None,
                  categorical_columns=None,
                  ordinal_columns=None,
                  continuous_columns=None,
                  preprocessor_eps: float = None,
                  nullable=False
                  ):

        columns = self.get_column_names(data)

        if meta_data is None:
            meta_data = {}
        if categorical_columns is None:
            categorical_columns = []
        if ordinal_columns is None:
            ordinal_columns = []
        if continuous_columns is None:
            continuous_columns = []

        def add_unique(s, str_list: list):
            if s not in str_list:
                str_list.append(s)

        if meta_data is not None:
            for meta_col in columns:
                if meta_col in meta_data.keys() and 'type' in meta_data[meta_col].keys():
                    type = meta_data[meta_col]['type']
                    if type == 'string':
                        add_unique(meta_col, categorical_columns)
                    if type == 'int':
                        add_unique(meta_col, ordinal_columns)
                    if type == 'float':
                        add_unique(meta_col, continuous_columns)

        if len(continuous_columns) + len(ordinal_columns) + len(categorical_columns) == 0:
            inferred = TypeMap.infer_column_types(data)
            categorical_columns = inferred['categorical_columns']
            ordinal_columns = inferred['ordinal_columns']
            continuous_columns = inferred['continuous_columns']
            if not nullable:
                nullable = len(inferred['nullable_columns']) > 0

        mapped_columns = categorical_columns + ordinal_columns + continuous_columns

        assert len(mapped_columns) == len(
            columns), 'Column mismatch. Make sure that the meta_data configuration defines all columns.'

        if transformer is None:
            col_tranformers = []
            for col in columns:
                if col in categorical_columns:
                    t = LabelTransformer(nullable=nullable)
                    col_tranformers.append(t)
                elif col in ordinal_columns:
                    lower = meta_data[col]['lower'] if col in meta_data and 'lower' in meta_data[col] else None
                    upper = meta_data[col]['upper'] if col in meta_data and 'upper' in meta_data[col] else None
                    t = OrdinalTransformer(lower=lower, upper=upper, nullable=nullable)
                    col_tranformers.append(t)
                elif col in continuous_columns:
                    lower = meta_data[col]['lower'] if col in meta_data and 'lower' in meta_data[col] else None
                    upper = meta_data[col]['upper'] if col in meta_data and 'upper' in meta_data[col] else None
                    t = MinMaxTransformer(lower=lower, upper=upper, nullable=nullable, negative=False)
                    col_tranformers.append(t)
            self._transformer = TableTransformer(col_tranformers)
        else:
            self._transformer = transformer

        train_data = self._get_train_data(
            data,
            style='NA',
            transformer=self._transformer,
            nullable=nullable,
            preprocessor_eps=preprocessor_eps,
            categorical_columns=categorical_columns,
            ordinal_columns=ordinal_columns,
            continuous_columns=continuous_columns
        )

        config = {}
        for col_tt, col in zip(self._transformer.transformers, columns):
            col_tt: ColumnTransformer
            if col in categorical_columns:
                cat_size = col_tt.cardinality[0]
                config[col] = {'type': 'string', 'size': cat_size}
            if col in ordinal_columns:
                ord_size = col_tt.cardinality[0]
                config[col] = {'type': 'int', 'size': ord_size}
            if col in continuous_columns:
                config[col] = {'type': 'float', 'size': 1}

        domain = Domain(config)
        data = Dataset(pd.DataFrame(np.array(train_data), columns=columns), domain)
        self.categorical_columns = categorical_columns
        self.ordinal_columns = ordinal_columns
        self.continuous_columns = continuous_columns

        return data
