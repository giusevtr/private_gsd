import pandas as pd
import numpy as np
from utils import MLEncoder
from experiments.utils import read_original_data, read_tabddpm_data
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier


import optuna
QUANTILES = 50

score_fn = make_scorer(f1_score, average='macro')
learning_rate = 0.1185499403580282
max_depth = 5
min_child_weight = 1
gamma = 0.1
subsample = 0.8407621684817781
reg_lambda = 8

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


def objective(trial):
    print('Trial:', trial.number)
    # Sample parameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1.0, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 2)
    gamma = trial.suggest_float("gamma", 0, 0.2)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    reg_lambda = trial.suggest_float("reg_lambda", 1, 10.0, log=True)

    scores = []

    # Train model and save validation score
    model = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth,
                          min_child_weight=min_child_weight, gamma=gamma, subsample=subsample, reg_lambda=reg_lambda)
    model.fit(X_train_oh, y_train)
    validation_score = score_fn(model, X_val_oh, y_val)
    scores.append(validation_score)

    ave_score = np.mean(scores)
    print(f'\tAverage validation score = {ave_score:.5f}')
    return ave_score


study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=0),
)

study.optimize(objective, n_trials=1000, show_progress_bar=False)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial
print(trial.number)
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
