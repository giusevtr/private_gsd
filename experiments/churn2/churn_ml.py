import pandas as pd
import numpy as np
from utils import MLEncoder
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier
from experiments.utils_for_experiment import read_original_data, read_tabddpm_data
import argparse
QUANTILES = 50
parser = argparse.ArgumentParser()
parser.add_argument('--sync_generator', default='gsd')
parser.add_argument('--ml_parameters', default='default')
args = parser.parse_args()

sync_gen = args.sync_generator
ml_parameters = args.sync_generator

print(f'Generator = {sync_gen}')
score_fn = make_scorer(f1_score, average='macro')
learning_rate = 0.1925742887404927
max_depth = 3
min_child_weight = 2
gamma = 0.17674635782779186
subsample = 0.638296403842424
reg_lambda = 0.410446516722773

get_ml_function = lambda rs: XGBClassifier(random_state=rs) if ml_parameters == 'default' else\
    XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight, gamma=gamma,
                  subsample=subsample, reg_lambda=reg_lambda, random_state=rs)


dataset_name = 'churn2'
k = 3
eps = 100000
data_size_str = 'N'

real_cols = [f'num_3', 'num_5', 'num_6']
ord_cols = ['num_0', 'num_1', 'num_2', 'num_4']

train_df, val_df, test_df, all_cols, cat_cols, num_cols = read_original_data(dataset_name)
all_df = pd.concat([train_df, val_df, test_df])

encoder = MLEncoder(cat_features=cat_cols, num_features=num_cols, target='Label', rescale=False)
encoder.fit(all_df)

X_train_oh, y_train = encoder.encode(train_df)
X_val_oh, y_val = encoder.encode(val_df)
X_test_oh, y_test = encoder.encode(test_df)

# Train a model on the original data and save test score
orig_scores_test = []
orig_scores_val = []
for rs in range(1):
    model = get_ml_function(rs)

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
    for rs in range(1):
        # Synthetic
        sync_df = None
        if sync_gen == 'gsd':
            sync_df = pd.read_csv(f'sync_data/{dataset_name}/3/100000.00/N/oneshot/sync_data_{seed}.csv').dropna()
        elif sync_gen == 'tabddpm':
            sync_df = read_tabddpm_data(dataset_name, seed=seed)

        X_sync_oh, y_sync = encoder.encode(sync_df)

        # Train a model on the synthetic data and save test score
        model_sync = get_ml_function(rs)

        model_sync.fit(X_sync_oh, y_sync)
        synthetic_train = score_fn(model_sync, X_sync_oh, y_sync)
        synthetic_val = score_fn(model_sync, X_val_oh, y_val)
        synthetic_test = score_fn(model_sync, X_test_oh, y_test)
        print(f'\tSynthetic:\t{synthetic_train:.5f}\t{synthetic_val:.5f}\t{synthetic_test:.5f}')
        scores_val.append(synthetic_val)
        scores_test.append(synthetic_test)

print(f'Synthetic Val:\tAverage={np.mean(scores_val):.5f}\tstd={np.std(scores_val):.5f}')
print(f'Synthetic Test:\tAverage={np.mean(scores_test):.5f}\tstd={np.std(scores_test):.5f}')
print(f'')






