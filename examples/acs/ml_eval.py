import os.path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from utils.utils_data import get_data
from sklearn.linear_model import LogisticRegression as Model

from sklearn.metrics import accuracy_score, make_scorer
scorer = make_scorer(accuracy_score)
import matplotlib.pyplot as plt

##########
data_name = f'folktables_2018_real_CA'
data_train = get_data(f'{data_name}-mixed-train',
                domain_name=f'domain/{data_name}-mixed', root_path='../../data_files/folktables_datasets_real')
domain = data_train.domain

data_name = f'folktables_2018_real_CA'
data_test = get_data(f'{data_name}-mixed-test',
                domain_name=f'domain/{data_name}-mixed', root_path='../../data_files/folktables_datasets_real')
df_test = data_test.df_real



rounds = 50

sync_paths = []
for eps in [0.07, 0.23, 0.52]:
    for r in [50, 75]:
        for seed in [0]:
            path = f'sync_data/PrivGAfast/Prefix/{data_name}/{eps:.2f}/{r}/sync_data_{seed}.csv'
            sync_paths.append(path)

acc_list = []
for path in sync_paths:
    if not os.path.exists(path): continue
    df_train = pd.read_csv(path)
    cat_cols = domain.get_categorical_cols()
    for cat in domain.get_categorical_cols():
        df_train[cat] = df_train[cat].round().astype(int)


    for target in ['PINCP']:
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
        acc_list.append(test_acc)

        print('\n#####')
        print(path)
        print(f'Target: {target}')
        print(f'Majority classifier test acc: {test_maj_acc}')
        print(f'Test acc: {train_acc}')
        print(f'Train acc: {test_acc}')
        print('#####\n')



plt.plot(range(0, len(acc_list)), acc_list, color='b', label='synthetic')
plt.hlines(xmin=0, xmax=rounds, y=acc_list[0], color='r', label='original')
plt.ylabel('Accuracy')
plt.xlabel('Halfpaces Adaptive Epoch')
plt.legend()
plt.show()