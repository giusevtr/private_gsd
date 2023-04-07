import itertools
import os.path
import pickle
import seaborn as sns

import jax.random
import matplotlib.pyplot as plt
import pandas as pd

from models import PrivGA, SimpleGAforSyncData
from stats import ChainedStatistics, Halfspace, Marginals
# from utils.utils_data import get_data
import jax.numpy as jnp
# from dp_data.data import get_data
from dp_data import load_domain_config, load_df, DataPreprocessor, ml_eval
from utils import timer, Dataset, Domain
from utils.cdp2adp import cdp_rho, cdp_eps
import numpy as np
from dev.ML.ml_utils import evaluate_machine_learning_task

from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":
    dataset_name = 'folktables_2018_multitask_CA'
    root_path = '../../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)
    df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
    df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')

    preprocesor: DataPreprocessor
    preprocesor = pickle.load(open(f'{root_path}/{dataset_name}/preprocessor.pkl', 'rb'))


    print(f'train size: {df_train.shape}')
    print(f'test size:  {df_test.shape}')

    domain = Domain.fromdict(config)
    cat_cols = domain.get_categorical_cols()
    num_cols = domain.get_numeric_cols()
    targets = ['JWMNP_bin', 'PINCP', 'ESR', 'MIG', 'PUBCOV']
    features = []
    for f in domain.attrs:
        if f not in targets:
            features.append(f)
    ml_fn = ml_eval.get_evaluate_ml(df_test, config, targets, models=['LogisticRegression'])

    data = Dataset(df_train, domain)
    # Debug marginals
    module0 = Marginals.get_kway_categorical(domain, k=2)
    stat_module = ChainedStatistics([module0])
    stat_module.fit(data)
    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module.get_dataset_statistics_fn()

    T = [50]
    S = [5]
    epsilon_vals = [1]
    seeds = [0, 1, 2]

    Res = []
    for eps, seed, t, s in itertools.product(epsilon_vals, seeds, T, S):

        sync_path = f'sync_data/RAP++/folktables_2018_multitask_CA/{eps:.2f}/({s}, {t})/{seed}/synthetic.csv'
        if not os.path.exists(sync_path):
            print(f'{sync_path} NOT FOUND')
            continue

        print(f'reading {sync_path}')
        df_sync_post = pd.read_csv(sync_path)

        # for cat in cat_cols:
        #     if cat == 'ESR': continue
        #     if cat != 'JWTR': continue
        #     # print(cat)
        #     df = df_train[['ESR', cat]].astype(int)
        #     df['Type'] = 'Real'
        #     df_sync = df_sync_post[['ESR', cat]].astype(int)
        #     df_sync['Type'] = 'Sync'
        #     df_concat = pd.concat([df, df_sync], ignore_index=True)
        #     df_concat[cat] = df_concat[cat].astype(str)
        #     # sns.histplot(data=df_concat, x=cat, hue='ESR', row='Type')
        #
        #     g = sns.FacetGrid(df_concat, row="Type", hue='ESR',
        #                       sharey=False)
        #     g.map_dataframe(sns.histplot, x=cat)
        #
        #
        #     plt.show()
        #     print('ploting cat=', cat)



        sync_dataset = Dataset(df_sync_post, domain)
        errors = np.abs(true_stats - stat_fn(sync_dataset))
        print(f'Marginal max error ={errors.max()}, mean error ={errors.mean()}')

        res = ml_fn(df_sync_post, seed=0)
        res = res[res['Eval Data'] == 'Test']
        res = res[res['Metric'] == 'f1_macro']
        print('seed=', seed, 'eps=', eps)
        print(res)
        for i, row in res.iterrows():
            target = row['target']
            f1 = row['Score']
            Res.append([dataset_name, 'Yes', 'RAP++', '2Cat+HS', t, s, 'LR', target, eps, 'F1', seed, f1])
            # Res.append([dataset_name, 'Yes', algo_name+query_name, 'LR', target, eps, 'Accuracy', seed, acc])

    results = pd.DataFrame(Res, columns=['Data', 'Is DP', 'Generator',
                                         'T', 'S',
                                         'Model',
                                         'Statistics',
                                         'Target', 'epsilon', 'Metric',
                                         'seed',
                                         'Score'])

    print(results)
    file_path = 'results'
    os.makedirs(file_path, exist_ok=True)
    file_path = f'results/results_rap++.csv'
    # if os.path.exists(file_path):
    #     results_pre = pd.read_csv(file_path, index_col=None)
    #     results = results_pre.append(results)
    print(f'Saving ', file_path)
    results.to_csv(file_path, index=False)