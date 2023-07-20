import os
import sys
import pickle

from sklearn.metrics import f1_score, accuracy_score

from catboost import CatBoostClassifier, Pool, cv

import pandas as pd
import numpy as np

import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
from utils import timer, Dataset, Domain
from eval_ml import  get_Xy, get_evaluate_ml

from xgboost import XGBClassifier

cat_only = True
# dataset_name = 'folktables_2018_travel_CA'
# dataset_name, target = ('folktables_2018_employment_CA', 'ESR')
# dataset_name, target = ('folktables_2018_income_CA', 'PINCP')
dataset_name, target = ('folktables_2018_travel_CA', 'JWMNP')
# dataset_name, target = ('folktables_2018_employment_CA', 'ESR')
N = '32000'
# root_path = 'conditional_3way/sync_data_copy/folktables_2018_travel_CA/3/100000.00/2000/oneshot'

# path = "sync_data/ACS Travel/2000/sync_data_0.csv"
seed = 0
print("running")

from dp_data import load_domain_config, load_df

root_path = '../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
config = load_domain_config(dataset_name, root_path=root_path)
domain = Domain(config=config)
df_train = load_df(dataset_name, root_path=root_path, idxs_path=f'seed{seed}/train')
df_test = load_df(dataset_name, root_path=root_path, idxs_path=f'seed{seed}/test')

sync_path = f'conditional_3way/sync_data_copy/{dataset_name}/3/100000.00/{N}/oneshot/sync_data_{seed}.csv'
df_sync = pd.read_csv(sync_path)

cat_cols = domain.get_categorical_cols() + domain.get_ordinal_cols()
if cat_only:
    domain = domain.project(cat_cols)

cat_cols.remove(target)
features = list(domain.attrs)
features.remove(target)

def cat_to_int(df, cat_features):
    convert_dict = {feat:'int32' for feat in cat_features}
    df = df.astype(convert_dict)
    return df


X_train, y_train, X_test, y_test = get_Xy(domain, target=target,
                                          df_train=df_sync, df_test=df_test,
                                          scale_real_valued=True)

model = XGBClassifier()
model.fit(X_train, y_train)

# r = eval_ml(df_sync, df_test, seed, group=None)
acc_on_df_sync = accuracy_score(y_true=y_train, y_pred=model.predict(X_train))
acc_on_df_test = accuracy_score(y_true=y_test, y_pred=model.predict(X_test))
print(f'Default parameters: {acc_on_df_sync}, {acc_on_df_test}')
print()
print()
print()

def objective(trial):

    model = XGBClassifier(
        learning_rate=trial.suggest_float("learning_rate", 0.05, 0.5, log=True),
        # n_estimators=trial.suggest_int("n_estimators", 100, 2000),
        max_depth=trial.suggest_int("max_depth", 3, 100),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 2),
        gamma=trial.suggest_float("gamma", 0, 0.2),
        subsample=trial.suggest_float("subsample", 0.5, 0.9),
        # colsample_bytree=0.8,
        # objective='binary:logistic',
        reg_alpha=trial.suggest_float("reg_alpha", 1e-5, 100.0),
    )
    model.fit(X_train, y_train)

    # r = eval_ml(df_sync, df_test, seed, group=None)
    acc_on_df_sync = accuracy_score(y_true=y_train, y_pred=model.predict(X_train))
    acc_on_df_test = accuracy_score(y_true=y_test, y_pred=model.predict(X_test))


    print('Trial:', trial.number)
    print('Params:', trial.params)
    print(f'Acc score on df_sync = {acc_on_df_sync}')
    print(f'Acc score on df_test = {acc_on_df_test}')

    scores = [acc_on_df_test]
    return np.mean(scores)


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