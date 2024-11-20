import os
import pandas as pd
from stats import ChainedStatistics, Prefix
import jax.numpy as jnp
from utils import timer, Dataset, Domain
from dp_data import load_domain_config, load_df
import jax


datasets = [# ACSReal
            # ('folktables_2018_real_CA', '3_84_2_5_0'),
            # ('folktables_2018_real_NY', '3_84_2_5_0'),
            # ('folktables_2018_real_TX', '3_84_2_5_0'),
            # other tasks
            ('folktables_2018_coverage_CA', '2_34_1_5_0'),
            ('folktables_2018_employment_CA', '2_16_1_5_0'),
            ('folktables_2018_income_CA', '2_18_1_5_0'),
            ('folktables_2018_mobility_CA', '2_68_1_5_0'),
            ('folktables_2018_travel_CA', '2_28_1_5_0'),
]

Res = []
for algo_name in ['RP', 'GN']:
    for (dataset_name, query_name) in datasets:

        root_path = '../../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
        config = load_domain_config(dataset_name, root_path=root_path)
        df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
        df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')

        domain = Domain.fromdict(config)
        data = Dataset(df_train, domain)
        binary_features = [(feat,) for feat in domain.get_categorical_cols() if domain.size(feat)==2]
        binary_size = sum([domain.size(feat) for feat in binary_features])
        MAX_QUERIES = 200000
        num_random_prefixes = MAX_QUERIES // binary_size
        module = Prefix(domain,
                        k_cat=1,
                        cat_kway_combinations=binary_features,
                        k_prefix=2,
                        num_random_prefixes=num_random_prefixes,
                        rng=jax.random.key(0))
        stat_module = ChainedStatistics([module])
        stat_module.fit(data)
        true_stats = stat_module.get_all_true_statistics()
        stat_fn = stat_module.get_dataset_statistics_fn()



        for seed in range(3):
            for eps in [1.0, 0.74, 0.52, 0.23, 0.07]:
                for T in [10, 20, 40, 60, 80]:
                    path = f'./rp_gn/saved_syndata/{dataset_name}/{query_name}/{algo_name}/{eps}/adaptive/0.5/{T}/syndata_{seed}.csv'
                    if not os.path.exists(path):
                        num_workloads = int(query_name.split('_')[1])
                        assert num_workloads < T, 'Error: missing a file that should exist.'
                        continue

                    print(f'reading {path}')
                    df_syndata = pd.read_csv(path)
                    sync_data = Dataset(df_syndata, domain)
                    errors = jnp.abs(true_stats - stat_fn(sync_data))
                    print(f'max error ={errors.max()}, mean = {errors.mean()}')

                    module_name = 'BT'
                    Res.append(
                        [algo_name, dataset_name, module_name, T, 1, eps, seed, 'Max', errors.max(),
                         0])
                    Res.append([algo_name, dataset_name, module_name, T, 1, eps, seed, 'Average', errors.mean(),
                                0])

columns = ['Generator', 'Data', 'Statistics', 'T', 'S', 'epsilon', 'seed', 'error type', 'error', 'time']
file_name = 'icml_results/rp_gn_results.csv'
results_df = pd.DataFrame(Res, columns=columns)
results_df.to_csv(file_name, index=False)
