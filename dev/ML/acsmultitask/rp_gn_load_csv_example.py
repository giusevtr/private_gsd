import os
import pandas as pd
import itertools
import os.path
import pickle

import jax.random
import matplotlib.pyplot as plt

from models import GSD, SimpleGAforSyncData
from stats import ChainedStatistics, Halfspace, Marginals
# from utils.utils_data import get_data
import jax.numpy as jnp
# from dp_data.data import get_data
from dp_data import load_domain_config, load_df, DataPreprocessor, ml_eval
from utils import timer, Dataset, Domain


dataset_name = 'folktables_2018_multitask_CA'
root_path = '../../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
config = load_domain_config(dataset_name, root_path=root_path)
df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')

preprocesor: DataPreprocessor
preprocesor = pickle.load(open(f'{root_path}/{dataset_name}/preprocessor.pkl', 'rb'))
domain = Domain.fromdict(config)
cat_cols = domain.get_categorical_cols()
num_cols = domain.get_numeric_cols()

print(f'train size: {df_train.shape}')
print(f'test size:  {df_test.shape}')
targets = ['JWMNP_bin', 'PINCP', 'ESR', 'MIG', 'PUBCOV']
features = []
for f in domain.attrs:
    if f not in targets:
        features.append(f)


# model = 'XGBoost'
model = 'LogisticRegression'
ml_fn = ml_eval.get_evaluate_ml(df_test, config, targets, models=[model])


datasets = [('folktables_2018_multitask_CA', '2_595_-1_5_0', 'BT'), # binary tree
            # ('folktables_2018_multitask_CA', 'label_2_130_-1_5_0', 'label_BT'), # binary tree, label only
            # ('folktables_2018_multitask_CA', '2_595_-1_hist_5_0', 'Hist'), # histogram
            # ('folktables_2018_multitask_CA', 'label_2_130_-1_hist_5_0', 'label_Hist'), # histogram, label only
            ]
Res = []
for algo_name in ['RP', 'GN']:
    for (dataset_name, query_code, query_name) in datasets:
        for seed in range(3):
            for eps in [1.0, 0.74, 0.52, 0.23, 0.07]:
                path = f'./sync_data/RAP_GEM/{dataset_name}/{query_code}/{algo_name}/{eps}/one_shot/10000/syndata_{seed}.csv'
                try:
                    df_syndata = pd.read_csv(path)
                    res = ml_fn(df_syndata, seed=0)
                    res = res[res['Eval Data'] == 'Test']
                    print(algo_name+query_name, 'seed=', seed, 'eps=', eps)
                    print(res)
                    for i, row in res.iterrows():
                        # target = row['target']
                        # f1 = row['Score']
                        # Res.append([dataset_name, 'Yes',  algo_name, query_name, 'oneshot', 'oneshot', model, target, eps, 'F1', seed, f1])
                        # Res.append([dataset_name, 'Yes', algo_name+query_name, 'LR', target, eps, 'Accuracy', seed, acc])
                        target = row['target']
                        metric = row['Metric']
                        score = row['Score']
                        Res.append([dataset_name, 'Yes', algo_name, query_name, 'oneshot', 'oneshot', model, target, eps,
                                    metric, seed, score])
                except:
                    print(f'Could not read', path)


results = pd.DataFrame(Res, columns=['Data', 'Is DP',
                                     'Generator',
                                     'Statistics',
                                     'T', 'S',
                                     'Model',
                                     'Target',
                                     'epsilon',
                                     'Metric',
                                     'seed',
                                     'Score'])

print(results)
os.makedirs('results_final', exist_ok=True)
file_path = f'results_final/results_rp_gem_{model}.csv'
# if os.path.exists(file_path):
#     results_pre = pd.read_csv(file_path, index_col=None)
#     results = results_pre.append(results)
print(f'Saving ', file_path)
results.to_csv(file_path, index=False)