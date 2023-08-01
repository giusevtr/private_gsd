import pandas as pd
import numpy as np
from utils import MLEncoder
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier
from experiments.utils import read_original_data, read_tabddpm_data
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--sync_generator', choices=['gsd', 'tabddpm'], default='gsd')
parser.add_argument('--ml_parameters', choices=['default', 'optimized'], default='default')
args = parser.parse_args()

sync_gen = args.sync_generator
ml_parameters = args.ml_parameters
print(ml_parameters)

dataset_name = 'miniboone'
# k = 2
stats_name = f'2_covar'
eps = 'inf'
data_size_str = '2000'

train_df, val_df, test_df, all_cols, cat_cols, num_cols = read_original_data(dataset_name)
all_df = pd.concat([train_df, val_df, test_df])

gsd_data_path = f'sync_data/{dataset_name}/{stats_name}/inf/{data_size_str}/oneshot/sync_data_{0}.csv'

sync_df = pd.read_csv(gsd_data_path).dropna()

train_df.pop('Label')
sync_df.pop('Label')
train_cov = train_df.cov().values / len(train_df)
sync_cov = sync_df.cov().values / 2000
err = np.abs(train_cov - sync_cov)
print(err.max())
print()

for i in range(50):
    col_name = f'num_{i}'
    print(col_name)
    # temp_df = train_df[train_df['Label'] == 1]
    print(f"{train_df[col_name].max():.4f}\t{sync_df[col_name].max():.4f}")
    print(f"{train_df[col_name].max():.4f}\t{sync_df[col_name].max():.4f}")
    print(f"{train_df[col_name].min():.4f}\t{sync_df[col_name].min():.4f}")
    print(f"{train_df[col_name].mean():.4f}\t{sync_df[col_name].mean():.4f}")
    print(f"{train_df[col_name].std():.4f}\t{sync_df[col_name].std():.4f}")
    print()
