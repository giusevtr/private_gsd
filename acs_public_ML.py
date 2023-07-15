import itertools
import os, sys
import jax.random
import pandas as pd
import numpy as np
from models import GeneticSDConsistent as GeneticSD
from models import GSD
from stats import ChainedStatistics,  Marginals, NullCounts
import jax.numpy as jnp
from dp_data import load_domain_config, load_df
from dp_data.data_preprocessor import DataPreprocessor
from utils import timer, Dataset, Domain


from eval_ml import  get_Xy, get_evaluate_ml


if __name__ == "__main__":

    cat_only = False
    Models = [
        # 'LogisticRegression',
        # 'RandomForest',
        'XGBoost'
    ]
    # epsilon = [0.07, 0.15, 0.23, 0.52, 0.74, 1, 100000]
    epsilon = [100000]
    # epsilon = [0.07, 0.15, 0.23, 0.52, 0.74, 1]
    data_size_str = 'N'
    k = 2
    SEEDS = [0, 1, 2]
    QUANTILES = 30

    private_state = 'CA'
    public_states = ['TX', 'NY', 'CA']

    DATA = [
        # ('folktables_2018_coverage', 'PUBCOV'),
        # ('folktables_2018_mobility', 'MIG'),
        # ('folktables_2018_employment', 'ESR'),
        # ('folktables_2018_income', 'PINCP'),
        ('folktables_2018_travel', 'JWMNP'),
    ]
    RESULTS = []
    for (dataset_name, target), public_state, model in itertools.product(DATA, public_states, Models):
    # for dataset_name, target in DATA:
    #     for public_state in public_states:
        priv_dataset_name = f'{dataset_name}_{private_state}'
        pub_dataset_name = f'{dataset_name}_{public_state}'

        root_path = 'dp-data-dev/datasets/preprocessed/folktables/1-Year/'
        config = load_domain_config(priv_dataset_name, root_path=root_path)
        df_train = load_df(priv_dataset_name, root_path=root_path, idxs_path='seed0/train')
        df_test = load_df(priv_dataset_name, root_path=root_path, idxs_path='seed0/test')

        config_public = load_domain_config(pub_dataset_name, root_path=root_path)
        df_train_public = load_df(pub_dataset_name, root_path=root_path, idxs_path='seed0/train')
        df_test_public = load_df(pub_dataset_name, root_path=root_path, idxs_path='seed0/test')

        bins_edges = {}
        quantiles = np.linspace(0, 1, QUANTILES)
        for att in config:
            if config_public[att]['type'] == 'numerical':
                v = df_train_public[att].values
                thresholds = np.quantile(v, q=quantiles)
                bins_edges[att] = thresholds
        domain = Domain(config=config, bin_edges=bins_edges)
        domain_public = Domain(config=config_public, bin_edges=bins_edges)
        if cat_only:
            cat_cols = domain.get_categorical_cols() + domain.get_ordinal_cols()
            domain = domain.project(cat_cols)
            domain_public = domain_public.project(cat_cols)

        data = Dataset(df_train, domain)
        N = len(data.df)
        data_size = N if data_size_str == 'N' else int(data_size_str)

        if data_size_str == 'N':
            data_size = len(df_train_public)
        else:
            data_size = int(data_size_str)
            df_train_public = df_train_public.sample(data_size)

        # df_train_public = df_train_public.sample(data_size)
        data_pub = Dataset(df_train_public, domain_public)
        public_data_size = len(df_train_public)

        eval_ml = get_evaluate_ml(
                                  domain=domain,
                                  targets=[target],
                                  models=[model],
                                  grid_search=False
                                  )
        res2 = eval_ml(df_train_public, df_test, 0, verbose=False)
        pub_f1 = res2[(res2['Eval Data'] == 'Test') & (res2['Metric'] == 'f1_macro')]['Score'].values[0]

        res1 = eval_ml(df_train, df_test, 0, verbose=False)
        real_f1 = res1[(res1['Eval Data'] == 'Test') & (res1['Metric'] == 'f1_macro')]['Score'].values[0]


        for eps in epsilon:
            print(f'Input data {dataset_name}, epsilon={eps:.2f}, data_size={data_size}, k={k} ')
            print(f'Public data state is {public_state}, with size {public_data_size}.')

            # Generate differentially private synthetic data with ADAPTIVE mechanism
            for seed in SEEDS:

                sync_dir = f'sync_data_public/{pub_dataset_name}/{priv_dataset_name}/{k}/{eps:.2f}/{data_size_str}/oneshot'
                sync_path = f'{sync_dir}/sync_data_{seed}.csv'
                if not os.path.exists(sync_path): continue
                sync_df = pd.read_csv(sync_path)
                print(f'Loading {sync_path}')

                sync_df = eval_ml(sync_df, df_test, 0, verbose=False)
                sync_f1 = sync_df[(sync_df['Eval Data'] == 'Test') & (sync_df['Metric'] == 'f1_macro')]['Score'].values[0]
                row_res = [eps, priv_dataset_name, pub_dataset_name,
                       seed, data_size_str, model, sync_f1, real_f1, pub_f1]

                print(row_res)
                RESULTS.append(row_res)

                results_df = pd.DataFrame(RESULTS, columns=[
                  'eps', 'Private Dataset', 'Public Dataset', 'Seed', 'Data size', 'Model', 'Sync f1',
                  'Real f1', 'Public f1'
                ])
                results_df.to_csv(f'public_data_results{"_cat" if cat_only else ""}.csv')

