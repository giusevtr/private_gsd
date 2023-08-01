
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, make_scorer
import optuna
from catboost import CatBoostClassifier

optuna.logging.set_verbosity(optuna.logging.ERROR)

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
    depth = trial.suggest_int("depth", 3, 10)
    l2_leaf_reg = trial.suggest_uniform("l2_leaf_reg", 0.1, 10.0)
    bagging_temperature = trial.suggest_uniform("bagging_temperature", 0.0, 1.0)
    leaf_estimation_iterations = trial.suggest_int("leaf_estimation_iterations", 1, 10)

    # Set parameters
    task_type = "GPU"
    iterations = 2000
    early_stop = 50
    od_pval = 0.001
    thread_count = 4


    # root_path = 'C:/Users/jncho/Documents/REUSE23/private_gsd/experiments/diabetes/data/diabetes/kfolds'
    root_path = '../../dp-data-dev/data2/data/diabetes/kfolds'
    for seed in range(5):
        # Load kfold data
        X_train = np.load(f'{root_path}/{seed}/X_num_train.npy')
        X_val = np.load(f'{root_path}/{seed}/X_num_val.npy')
        y_train = np.load(f'{root_path}/{seed}/y_train.npy')
        y_val = np.load(f'{root_path}/{seed}/y_val.npy')

        # Train model and save validation score
        model = CatBoostClassifier(learning_rate=learning_rate, depth=depth,
                              l2_leaf_reg = l2_leaf_reg, bagging_temperature = bagging_temperature, 
                              leaf_estimation_iterations = leaf_estimation_iterations, task_type = task_type, 
                              iterations = iterations, early_stopping_rounds = early_stop, od_pval = od_pval, thread_count = thread_count,
                              verbose = False)
        model.fit(X_train, y_train)
        validation_score = score_fn(model, X_val, y_val)
        print(validation_score)
        scores.append(validation_score)

    ave_score = np.mean(scores)
    print(f'\tAverage validation score = {ave_score:.5f}')
    return ave_score


study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=0),
)


study.optimize(objective, n_trials=1000, show_progress_bar=True)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial
print(trial.number)
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
