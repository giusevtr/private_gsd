import itertools
import os.path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from utils.utils_data import get_data
from sklearn.linear_model import LogisticRegression as Model

from sklearn.metrics import accuracy_score, make_scorer
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

    model = Model()
    model.fit(X_train, y_train)

    train_acc = scorer(model, X_train, y_train)
    test_acc = scorer(model, X_test, y_test)

    # print('\n#####')
    # print(f'Target: {target}')
    # print(f'Majority classifier test acc: {test_maj_acc}')
    # print(f'Test acc: {train_acc}')
    # print(f'Train acc: {test_acc}')
    # print('#####\n')
    return train_acc, test_acc



##########
data_name = f'folktables_2018_real_CA'
data_train = get_data(f'{data_name}-mixed-train',
                domain_name=f'domain/{data_name}-mixed', root_path='data_files/folktables_datasets_real')
domain = data_train.domain
df_train = data_train.df

data_name = f'folktables_2018_real_CA'
data_test = get_data(f'{data_name}-mixed-test',
                domain_name=f'domain/{data_name}-mixed', root_path='data_files/folktables_datasets_real')
df_test = data_test.df


test_acc_original = {}
for target in domain.get_categorical_cols():
    train_acc, test_acc = linear_ml_accuracy(df_train, df_test, target)
    test_acc_original[target] = test_acc

rounds = 50

sync_paths = []
acc_list = []

Results = []

algo = [
    'PrivGA',
    # 'GEM',
    # 'RAP'
]
queries = [
    'Halfspaces',
    'Ranges'
]
for a, q in itertools.product(algo, queries):
    for T in [25, 50, 75, 100, 10, 20, 40, 60, 80]:
        for eps in [0.07, 0.23, 0.52, 0.74, 1.00]:
            for seed in [0, 1, 2]:
                path = f'sync_data/{a}/{q}/{T:03}/{eps:.2f}/sync_data_{seed}.csv'
                if not os.path.exists(path): continue
                print(f'reading {path}')
                sync_paths.append(path)
                df_train_sync = pd.read_csv(path)
                if 'Unnamed: 0' in df_train_sync.columns:
                    df_train_sync.drop('Unnamed: 0', axis=1, inplace=True)

                cat_cols = domain.get_categorical_cols()
                for cat in domain.get_categorical_cols():
                    df_train_sync[cat] = df_train_sync[cat].round().astype(int)

                for target in cat_cols:
                    train_acc, test_acc = linear_ml_accuracy(df_train_sync, df_test, target)
                    acc_list.append(test_acc)

                    Results.append([f'{a}({q})', T, eps, seed, f'{target} ML Acc', test_acc_original[target] - test_acc])
                    print(path, f'\ntarget ={target}: Original acc = {test_acc_original[target]:.5f},'
                                f' Synthetic acc = {test_acc:.5f}')



cols = ['algo', 'T', 'epsilon', 'seed', 'error type', 'error']
df_results = pd.DataFrame(Results, columns=cols)

df_results = df_results.groupby(['algo', 'error type', 'T', 'epsilon'], as_index=False)['error'].mean()
df_results = df_results.groupby(['algo', 'error type', 'epsilon'], as_index=False)['error'].min()

sns.relplot(data=df_results, x='epsilon', y='error', col='error type', hue='algo', kind='line')
# plt.plot(range(0, len(acc_list)), acc_list, color='b', label='synthetic')
# plt.hlines(xmin=0, xmax=rounds, y=acc_list[0], color='r', label='original')
plt.ylabel('Error')
plt.xlabel('Halfpaces Adaptive Epoch')
plt.show()