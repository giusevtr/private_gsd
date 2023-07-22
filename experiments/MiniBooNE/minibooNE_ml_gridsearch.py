import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, make_scorer
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import KFold

# Create a score function using F1-macro
score_fn = make_scorer(f1_score, average='macro')

# ordinal = []
# columns = []
# target = 'Label'
# features = list(columns)
# features.remove(target)


def objective(trial):
    print('Trial:', trial.number)
    scores = []
    # Sample parameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1.0, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 2)
    gamma = trial.suggest_float("gamma", 0, 0.2)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    reg_lambda = trial.suggest_float("reg_lambda", 1, 10.0, log=True)

    idx_train = np.load(f'../../dp-data-dev/data2/data/adult/idx_train.npy')
    X_train_num = np.load(f'../../dp-data-dev/data2/data/adult/X_num_train.npy')
    X_train_cat = np.load(f'../../dp-data-dev/data2/data/adult/X_cat_train.npy')
    X_train = np.column_stack((X_train_cat, X_train_num))
    y_train = np.load(f'../../dp-data-dev/data2/data/adult/y_train.npy')

    X_val_num = np.load(f'../../dp-data-dev/data2/data/adult/X_num_val.npy')
    X_val_cat = np.load(f'../../dp-data-dev/data2/data/adult/X_cat_val.npy')
    X_val = np.column_stack((X_val_cat, X_val_num))
    y_val = np.load(f'../../dp-data-dev/data2/data/adult/y_val.npy')

    # kfold = KFold(3)
    scores = []

    # Train model and save validation score
    model = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth,
                          min_child_weight=min_child_weight, gamma=gamma, subsample=subsample, reg_lambda=reg_lambda)
    model.fit(X_train, y_train)
    validation_score = score_fn(model, X_val, y_val)
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
