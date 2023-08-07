import pandas as pd
import os
from models import GSD
from utils import Dataset, Domain
import numpy as np

import jax.random
from models import GeneticSDConsistent as GeneticSD
from models import GSD
from stats import ChainedStatistics,  Marginals, NullCounts
from utils import MLEncoder
import jax.numpy as jnp
import matplotlib.pyplot as plt

from stats.thresholds import get_thresholds_ordinal

from stats import ChainedStatistics,  Marginals, NullCounts
from dp_data import cleanup, DataPreprocessor, get_config_from_json
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier

import seaborn as sns
QUANTILES = 50

score_fn = make_scorer(f1_score, average='macro')
learning_rate = 0.1185499403580282
max_depth = 7
min_child_weight = 1
gamma = 0.16393189784441567
subsample = 0.8407621684817781
reg_lambda = 2

data_name = 'adult'
k = 2
data_size_str = '1000'
data_path = '../data2/data'
X_num_train = np.load(f'{data_path}/{data_name}/X_num_train.npy').astype(int)
X_num_val = np.load(  f'{data_path}/{data_name}/X_num_val.npy').astype(int)
X_num_test = np.load( f'{data_path}/{data_name}/X_num_test.npy').astype(int)
X_cat_train = np.load(f'{data_path}/{data_name}/X_cat_train.npy')
X_cat_val = np.load(  f'{data_path}/{data_name}/X_cat_val.npy')
X_cat_test = np.load( f'{data_path}/{data_name}/X_cat_test.npy')
y_train = np.load(    f'{data_path}/{data_name}/y_train.npy')
y_val = np.load(      f'{data_path}/{data_name}/y_val.npy')
y_test = np.load(     f'{data_path}/{data_name}/y_test.npy')

cat_cols = [f'cat_{i}' for i in range(X_cat_train.shape[1])]
num_cols = [f'num_{i}' for i in range(X_num_train.shape[1])]
all_cols = cat_cols + num_cols + ['Label']

# X_train = np.column_stack((X_cat_train, X_num_train))
# X_val = np.column_stack((X_cat_val, X_num_val))
# X_test = np.column_stack((X_cat_test, X_num_test))

train_df = pd.DataFrame(np.column_stack((X_cat_train, X_num_train, y_train)), columns=all_cols)
val_df = pd.DataFrame(np.column_stack((X_cat_val, X_num_val, y_val)), columns=all_cols)
test_df = pd.DataFrame(np.column_stack((X_cat_test, X_num_test, y_test)), columns=all_cols)
all_df = pd.concat([train_df, val_df, test_df])

encoder = MLEncoder(cat_features=cat_cols, num_features=num_cols, target='Label', rescale=False)
encoder.fit(all_df)

X_train_oh, y_train = encoder.encode(train_df)
X_val_oh, y_val = encoder.encode(val_df)
X_test_oh, y_test = encoder.encode(test_df)

protected_feat = 'cat_5'
orig_scores_test = []
orig_scores_val = []
for rs in range(1):
    # Train a model on the original data and save test score
    model = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight,
                          gamma=gamma, subsample=subsample, reg_lambda=reg_lambda, random_state=rs)

    model.fit(X_train_oh, y_train)
    original_train = score_fn(model, X_train_oh, y_train)
    original_val = score_fn(model, X_val_oh, y_val)
    original_test = score_fn(model, X_test_oh, y_test)
    print(f'\tOriginal:\t{original_train:.5f}\t{original_val:.5f}\t{original_test:.5f}')
    groups = test_df[protected_feat].unique()
    for g in groups:
        group_df = test_df[test_df[protected_feat] == g]
        X_sub_oh, y_sub = encoder.encode(group_df)
        subgroup_test = score_fn(model, X_sub_oh, y_sub)
        print(f'\t\tgroup={g:<20}\t\t{subgroup_test:.5f}')

    orig_scores_test.append(original_test)
    orig_scores_val.append(original_val)

g = sns.FacetGrid(train_df[[protected_feat, 'Label']], col=protected_feat)
g.map(sns.histplot, 'Label', stat='density')
plt.show()
# print(f'Original Val:\tAverage={np.mean(orig_scores_val):.5f}\tstd={np.std(orig_scores_val):.5f}')
# print(f'Original Test:\tAverage={np.mean(orig_scores_test):.5f}\tstd={np.std(orig_scores_test):.5f}')
print()
print()

scores_test = []
scores_val = []

groups = test_df[protected_feat].unique()
group_val = dict([(g, []) for g in groups])

for seed in [0]:
    # Synthetic
    # sync_df = pd.read_csv(f'sync_data_race/adult/4/100000.00/N/oneshot/sync_data_{seed}.csv').dropna()
    sync_df = pd.read_csv(f'sync_data/adult/{k}/inf/{data_size_str}/oneshot/sync_data_{seed}_old.csv').dropna()
    X_sync_oh, y_sync = encoder.encode(sync_df)

    for rs in range(5):

        # Train a model on the synthetic data and save test score
        model_sync = XGBClassifier(
            learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight,
                                  gamma=gamma, subsample=subsample,
            reg_lambda=reg_lambda)
        model_sync.fit(X_sync_oh, y_sync)
        synthetic_train = score_fn(model_sync, X_sync_oh, y_sync)
        synthetic_val = score_fn(model_sync, X_val_oh, y_val)
        synthetic_test = score_fn(model_sync, X_test_oh, y_test)
        print(f'Synthetic:\t{synthetic_train:.5f}\t{synthetic_val:.5f}\t{synthetic_test:.5f}')

        for g in groups:
            group_df = test_df[test_df[protected_feat] == g]
            X_sub_oh, y_sub = encoder.encode(group_df)
            subgroup_test = score_fn(model_sync, X_sub_oh, y_sub)
            group_val[g].append(subgroup_test)
            print(f'\t\tgroup={g:<20}\t\t{subgroup_test:.5f}')

        scores_val.append(synthetic_val)
        scores_test.append(synthetic_test)

print(f'Synthetic Val:\tAverage={np.mean(scores_val):.5f}\tstd={np.std(scores_val):.5f}')
print(f'Synthetic Test:\tAverage={np.mean(scores_test):.5f}\tstd={np.std(scores_test):.5f}')
for g in groups:
    print(f'\t\tgroup={g:<20}\t\tAverage={np.mean(group_val[g]):.5f}')






