
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, make_scorer
import optuna
from xgboost import XGBClassifier


# Create a score function using F3-macro
score_fn = make_scorer(f1_score, average='macro')

ordinal = ['Pregnant', 'Plasma',   'BloodPressure',  'Triceps', 'Age', 'Label']
columns = ['Pregnant',# 17
           'Plasma',# 199
           'BloodPressure', # 122
           'Triceps',# 99
           'Insulin', # 846
           'BMI', # 67.1
           'Diabetes_pedigree', # 2.342
           'Age', # 60
           'Label']
target = 'Label'
features = list(columns)
features.remove(target)


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
    for seed in range(5):
        # Load kfold data
        X_train = np.load(f'dp-data-dev/data2/data/diabetes/kfolds/{seed}/X_num_train.npy')
        X_val = np.load(f'dp-data-dev/data2/data/diabetes/kfolds/{seed}/X_num_val.npy')
        y_train = np.load(f'dp-data-dev/data2/data/diabetes/kfolds/{seed}/y_train.npy')
        y_val = np.load(f'dp-data-dev/data2/data/diabetes/kfolds/{seed}/y_val.npy')

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
