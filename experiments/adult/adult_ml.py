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


QUANTILES = 50

score_fn = make_scorer(f1_score, average='macro')
learning_rate = 0.1185499403580282
max_depth = 7
min_child_weight = 1
gamma = 0.16393189784441567
subsample = 0.8407621684817781
reg_lambda = 2

X_num_train = np.load(f'../../dp-data-dev/data2/data/adult/X_num_train.npy').astype(int)
X_num_val = np.load(f'../../dp-data-dev/data2/data/adult/X_num_val.npy').astype(int)
X_num_test = np.load(f'../../dp-data-dev/data2/data/adult/X_num_test.npy').astype(int)

X_cat_train = np.load(f'../../dp-data-dev/data2/data/adult/X_cat_train.npy')
X_cat_val = np.load(f'../../dp-data-dev/data2/data/adult/X_cat_val.npy')
X_cat_test = np.load(f'../../dp-data-dev/data2/data/adult/X_cat_test.npy')

y_train = np.load(f'../../dp-data-dev/data2/data/adult/y_train.npy').astype(int)
y_val = np.load(f'../../dp-data-dev/data2/data/adult/y_val.npy').astype(int)
y_test = np.load(f'../../dp-data-dev/data2/data/adult/y_test.npy').astype(int)

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


orig_scores_test = []
orig_scores_val = []
for rs in range(10):
    # Train a model on the original data and save test score
    model = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight,
                          gamma=gamma, subsample=subsample, reg_lambda=reg_lambda, random_state=rs)

    model.fit(X_train_oh, y_train)
    original_train = score_fn(model, X_train_oh, y_train)
    original_val = score_fn(model, X_val_oh, y_val)
    original_test = score_fn(model, X_test_oh, y_test)
    print(f'\tOriginal:\t{original_train:.5f}\t{original_val:.5f}\t{original_test:.5f}')

    orig_scores_test.append(original_test)
    orig_scores_val.append(original_val)


print(f'Original Val:\tAverage={np.mean(orig_scores_val):.5f}\tstd={np.std(orig_scores_val):.5f}')
print(f'Original Test:\tAverage={np.mean(orig_scores_test):.5f}\tstd={np.std(orig_scores_test):.5f}')

scores_test = []
scores_val = []
for seed in [0, 1, 2, 3]:
    # Synthetic
    sync_df = pd.read_csv(f'sync_data/adult/3/100000.00/N/oneshot/sync_data_{seed}.csv').dropna()
    X_sync_oh, y_sync = encoder.encode(sync_df)

    for rs in range(10):

        # Train a model on the synthetic data and save test score
        model_sync = XGBClassifier(
            # learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight,
            #                       gamma=gamma, subsample=subsample,
            reg_lambda=reg_lambda)
        model_sync.fit(X_sync_oh, y_sync)
        synthetic_train = score_fn(model_sync, X_sync_oh, y_sync)
        synthetic_val = score_fn(model_sync, X_val_oh, y_val)
        synthetic_test = score_fn(model_sync, X_test_oh, y_test)
        print(f'Synthetic:\t{synthetic_train:.5f}\t{synthetic_val:.5f}\t{synthetic_test:.5f}')
        scores_val.append(synthetic_val)
        scores_test.append(synthetic_test)

print(f'Synthetic Val:\tAverage={np.mean(scores_val):.5f}\tstd={np.std(scores_val):.5f}')
print(f'Synthetic Test:\tAverage={np.mean(scores_test):.5f}\tstd={np.std(scores_test):.5f}')






