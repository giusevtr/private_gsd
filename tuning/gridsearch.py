import os
import sys
import pickle

from sklearn.metrics import f1_score

from catboost import CatBoostClassifier, Pool, cv

import pandas as pd
import numpy as np

import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
from utils import timer, Dataset, Domain

cat_only = True
dataset_name = 'folktables_2018_travel_CA'
N = '2000'
# root_path = 'conditional_3way/sync_data_copy/folktables_2018_travel_CA/3/100000.00/2000/oneshot'

# path = "sync_data/ACS Travel/2000/sync_data_0.csv"
seed = 0
target = "JWMNP"
print("running")

from dp_data import load_domain_config, load_df

root_path = '../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
config = load_domain_config(dataset_name, root_path=root_path)
domain = Domain(config=config)
df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')

sync_path = f'conditional_3way/sync_data_copy/folktables_2018_travel_CA/3/100000.00/{N}/oneshot/sync_data_0.csv'
df_sync = pd.read_csv(sync_path)

if cat_only:
    cat_cols = domain.get_categorical_cols() + domain.get_ordinal_cols()
    domain = domain.project(cat_cols)

features = list(domain.attrs)
features.remove(target)

#
# def get_data(path):
#     data = pd.read_csv(path)
#     return data

def cat_to_int(df, cat_features):
    convert_dict = {feat:'int32' for feat in cat_features}
    df = df.astype(convert_dict)
    return df


x = cat_to_int(df_sync[features], features)
y = df_sync[target]

x_test1 = cat_to_int(df_train[features], features)
y_test1 = df_train[target]
x_test2 = cat_to_int(df_test[features], features)
y_test2 = df_test[target]

train_pool = Pool(x, y, cat_features=features)
test_pool1 = Pool(x_test1, y_test1, cat_features=features)
test_pool2 = Pool(x_test2, y_test2, cat_features=features)


def objective(trial):

    scores = []
    model = CatBoostClassifier(
        learning_rate=trial.suggest_float("learning_rate", 0.001, 1.0, log=True),
        depth=trial.suggest_int("depth", 3, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.1, 10.0),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
        leaf_estimation_iterations=trial.suggest_int("leaf_estimation_iterations", 1, 10),
        task_type="GPU",
        silent=True,
    )

    model.fit(train_pool)

    f1_on_df_sync = f1_score(y_true=y, y_pred=model.predict(train_pool), average="macro")
    f1_on_df_train = f1_score(y_true=y_test1, y_pred=model.predict(test_pool1), average="macro")
    f1_on_df_test = f1_score(y_true=y_test2, y_pred=model.predict(test_pool2), average="macro")
    print()
    print(f'F1 score on df_sync = {f1_on_df_sync}')
    print(f'F1 score on df_train = {f1_on_df_train}')
    print(f'F1 score on df_test = {f1_on_df_test}')
    scores.append(f1_on_df_test)

    return np.mean(scores)


study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=0),
)

study.optimize(objective, n_trials=100, show_progress_bar=False)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))