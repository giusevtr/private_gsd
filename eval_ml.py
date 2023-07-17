import argparse
import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, average_precision_score, accuracy_score
from typing import Tuple, Callable
from utils import Domain, Dataset
from dp_data.ml_models import MODELS, MODEL_PARAMS
import matplotlib.pyplot as plt


def separate_cat_and_num_cols(domain: Domain):
    # train_cols = [c for c in domain.attrs if c != target]
    train_cols_num = domain.get_numerical_cols()
    train_cols_cat = domain.get_categorical_cols()+ domain.get_ordinal_cols()
    return train_cols_num, train_cols_cat

def get_Xy(domain: Domain, target, df_train: pd.DataFrame, df_test: pd.DataFrame,
           scale_real_valued=True):
    cols_num, cols_cat = separate_cat_and_num_cols(domain)
    if target in cols_num: cols_num.remove(target)
    if target in cols_cat: cols_cat.remove(target)

    y_train = df_train[target].values
    y_test = df_test[target].values
    X_train = None
    X_test = None

    if len(cols_cat) > 0:
        X_cat_train = df_train[cols_cat].values
        X_cat_test = df_test[cols_cat].values
        X = np.concatenate((X_cat_train, X_cat_test))
        # categories = [np.arange(domain[c]['size']) for c in cols_cat]
        categories = [np.unique(np.concatenate((df_train[c].values, df_test[c].values))) for c in cols_cat]
        enc = OneHotEncoder(categories=categories)
        enc.fit(X)
        X_cat_train = enc.transform(X_cat_train).toarray()
        X_cat_test = enc.transform(X_cat_test).toarray()
        X_train = X_cat_train
        X_test = X_cat_test

    if len(cols_num) > 0:
        X_num_train = df_train[cols_num].values
        X_num_test = df_test[cols_num].values
        if scale_real_valued:
            scaler = StandardScaler()
            scaler.fit(X_num_train)
            X_num_train = scaler.transform(X_num_train)
            X_num_test = scaler.transform(X_num_test)

        if X_train is not None:
            X_train = np.concatenate((X_train, X_num_train), axis=1)
            X_test = np.concatenate((X_test, X_num_test), axis=1)
        else:
            X_train = X_num_train
            X_test = X_num_test

    assert X_train is not None
    assert X_test is not None
    return X_train, y_train, X_test, y_test



def fi(X_train):
    feature_names = [f'f {i}' for i in range(X_train.shape[1])]
    # importances = model.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    # forest_importances = pd.Series(importances, index=feature_names)
    # fig, ax = plt.subplots()
    # forest_importances.plot.bar(yerr=std, ax=ax)
    # ax.set_title("Feature importances using MDI")
    # ax.set_ylabel("Mean decrease in impurity")
    # fig.tight_layout()
    # plt.show()

def get_evaluate_ml(
        domain,
        targets: list,
        models: list,
        grid_search: bool = False,
        rescale=True
) -> Callable:
    features = []
    for f in domain.attrs:
        if f not in targets:
            features.append(f)

    models = MODELS.keys() if models is None else models

    def eval_fn(df_train, df_test, seed=0, group=None, verbose: bool=False):


        res = []
        for target in targets:
            if domain.size(target) > 2: continue
            f1_scoring = 'f1_macro'
            scorers = {}
            if f1_scoring == 'f1':
                scorers[f1_scoring] = make_scorer(f1_score)
            else:
                scorers[f1_scoring] = make_scorer(f1_score, average='macro')
            scorers['accuracy'] = make_scorer(accuracy_score)

            X_train, y_train, X_test, y_test = get_Xy(domain, target=target,
                                                          df_train=df_train, df_test=df_test,
                                                          scale_real_valued=rescale)
            for model_name in models:
                model = MODELS[model_name]
                model.random_state = seed

                import time
                start_time = time.time()

                if grid_search:
                    params = MODEL_PARAMS[model_name]
                    gridsearch = GridSearchCV(model, param_grid=params, cv=3, scoring=f1_scoring, verbose=3)
                    gridsearch.fit(X_test, y_test)
                    model = gridsearch.best_estimator_
                    if verbose: print(f'Best parameters: {gridsearch.best_params_}')
                else:
                    model.fit(X_train, y_train)

                for metric_name, scorer in scorers.items():
                    metric_train = scorer(model, X_train, y_train)
                    metric_test = scorer(model, X_test, y_test)
                    if verbose: print(f'\tModel {model_name}')
                    if verbose: print(f'\tTrain {metric_name}: {metric_train}')
                    if verbose: print(f'\tTest {metric_name}: {metric_test}')
                    res.append([model_name, target, 'Train', metric_name, metric_train, None])
                    res.append([model_name, target, 'Test', metric_name, metric_test, None])

                if group is not None:
                    group_values = np.unique(df_test[group].values)
                    for g in group_values:
                        mask = np.array(df_test[group] == g)
                        sub_idx = np.argwhere(mask).flatten()
                        X_test_sub = X_test[sub_idx, :]
                        y_test_sub = y_test[sub_idx]
                        size = sub_idx.shape[0]

                        mask_train = np.array(df_train[group] == g)
                        sub_idx_train = np.argwhere(mask_train).flatten()
                        y_train_sub = y_train[sub_idx_train]
                        size_train = sub_idx_train.shape[0]

                        # X_test_sub = X_test.

                        if verbose:
                            print(f'Group({group}) = {g}. ')
                            print(f'\tGroup size={size}.')
                            y_test_sum = np.sum(y_test_sub)
                            y_train_sum = np.sum(y_train_sub)
                            print(f'\tNumber of positive Test  samples in the group: {y_test_sum}.'
                                  f'Test group size = {size}. ({y_test_sum / size})')
                            print(f'\tNumber of positive Train samples in the group: {y_train_sum}.'
                                  f'Train group size = {size_train}. ({y_train_sum / size_train})')

                        for metric_name, scorer in scorers.items():
                            metric_train = scorer(model, X_train, y_train)
                            metric_test = scorer(model, X_test_sub, y_test_sub)
                            if verbose: print(f'\tTrain {metric_name}: {metric_train}')
                            if verbose: print(f'\tTest {metric_name}: {metric_test}')
                            res.append([model_name, target, 'Train', metric_name, metric_train, g])
                            res.append([model_name, target, 'Test', metric_name, metric_test, g])

                        end_time = time.time()
                        if verbose: print(f'Total time (s): {end_time - start_time}')
        df1 = pd.DataFrame(res, columns=['Model', 'target', 'Eval Data', 'Metric', 'Score', 'Sub Score'])
        return df1

    return eval_fn


