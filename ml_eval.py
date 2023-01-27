import itertools
import os.path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from utils import Dataset

from utils.utils_data import get_data
# from sklearn.linear_model import LogisticRegression as Model
from sklearn.ensemble import RandomForestClassifier as Model
from sklearn.metrics import accuracy_score, make_scorer
# scorer = make_scorer(accuracy_score)
from sklearn.metrics import accuracy_score, f1_score
scorer = make_scorer(accuracy_score)

import matplotlib.pyplot as plt
import seaborn as sns


def linear_ml_accuracy(df_train, df_test, target='PINCP'):
    train_cols = [c for c in df_train.columns if c != target]
    train_cols_num = [c for c in train_cols if domain[c] == 1]
    train_cols_cat = [c for c in train_cols if c not in train_cols_num]

    y_train, y_test = df_train[target].values, df_test[target].values
    from scipy.stats import mode
    maj_pred = mode(y_train)[0][0]
    test_maj_acc = (y_test == maj_pred).mean()

    X_train, X_test = df_train[train_cols_num].values, df_test[train_cols_num].values
    X_train_cat, X_test_cat = df_train[train_cols_cat].values, df_test[train_cols_cat].values

    categories = [np.arange(domain[c]) for c in train_cols_cat]
    enc = OneHotEncoder(categories=categories)

    enc.fit(X_train_cat)
    X_train_cat = enc.transform(X_train_cat).toarray()
    X_test_cat = enc.transform(X_test_cat).toarray()

    X_train = np.concatenate((X_train, X_train_cat), axis=1)
    X_test = np.concatenate((X_test, X_test_cat), axis=1)

    # model = Model(n_estimators=500, max_depth=7)
    model = Model()
    model.fit(X_train, y_train)

    train_acc = scorer(model, X_train, y_train)
    test_acc = scorer(model, X_test, y_test)

    print('\n#####')
    print(f'Target: {target}')

    print(f'Train labels')
    print(df_train[target].value_counts(normalize=True))
    print(f'Test labels')
    print(df_test[target].value_counts(normalize=True))
    print(f'Majority classifier test acc: {test_maj_acc}')
    print(f'Train error: {1-train_acc}')
    print(f'Test error: {1-test_acc}')
    print('#####\n')
    return train_acc, test_acc



##########
data_name = f'folktables_2018_real_CA'
data_train = get_data(f'{data_name}-mixed-train',
                domain_name=f'domain/{data_name}-mixed', root_path='data_files/folktables_datasets_real')
domain = data_train.domain
# df_train = data_train.df

data_name = f'folktables_2018_real_CA'
data_test = get_data(f'{data_name}-mixed-test',
                domain_name=f'domain/{data_name}-mixed', root_path='data_files/folktables_datasets_real')
df_test = data_test.df


# df_train = pd.read_csv('sync_data/PrivGA/Prefix/100/1.00/sync_data_0.csv')
# df_train = pd.read_csv('sync_data/PrivGA/Prefix/100/1.00/sync_data_2.csv')
# df_train = pd.read_csv('sync_data/RAP/Ranges/080/1.00/sync_data_0.csv')
# df_train = pd.read_csv('sync_data/RAP/Ranges/040/0.07/sync_data_0.csv')

df_train = Dataset.synthetic(data_train.domain, N=10000, seed=0).df
# for ncol in domain.get_numeric_cols():
#     sync = df_train[[ncol]]
#     sync['type'] = 'Sync'
#     real = df_test[[ncol]]
#     real['type'] = 'Real'
#
#     df = pd.concat([sync, real], ignore_index=True)
#     print(df)
#     df.hist(by='type', sharex=True)
#     plt.title(f'col ={ncol}')
#     plt.show()

test_acc_original = {}
for target in domain.get_categorical_cols():
    print()
    # print(df_train[target].value_counts(normalize=True))
    train_acc, test_acc = linear_ml_accuracy(df_train, df_test, target)
    test_acc_original[target] = test_acc
    # print(f'target={target}: train_error={1-train_acc:.5f}, test_error = {1-test_acc:.5f}')

