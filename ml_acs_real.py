import itertools
import os.path
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from utils.utils_data import get_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score, make_scorer, f1_score
scorer = make_scorer(accuracy_score)


def linear_ml_accuracy(df_train, df_test, target='PINCP', model_name='XGBoost'):
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
    if model_name == 'LR':
        model = LogisticRegression(random_state=0, max_iter=1000)
    elif model_name == 'RF':
        model = RandomForestClassifier(random_state=0)
    elif model_name == 'XGBoost':
        model = XGBClassifier(random_state=0)
    else:
        lf = LogisticRegression(random_state=0, max_iter=1000)
        rf = RandomForestClassifier(random_state=0)
        xgb = XGBClassifier(random_state=0)
        estimators = [('lf', lf), ('rf', rf), ('xgb', xgb)]
        model = VotingClassifier(estimators, voting='hard')

    model.fit(X_train, y_train)

    train_acc = scorer(model, X_train, y_train)
    test_acc = scorer(model, X_test, y_test)

    print('\n#####')
    print(f'Target: {target}')
    print(f'Majority classifier test acc: {test_maj_acc}')
    print(f'Train acc: {train_acc}')
    print(f'Test acc: {test_acc}')
    print('#####\n')
    return train_acc, test_acc


model_name = 'LR'  # you could modify this for different models
algo = [
    'PrivGA',
    'GEM',
    'RAP',
    'RAP++'
]
queries = [
    'Halfspaces',
    'Prefix',
    'Ranges'
]
##########
data_name = f'folktables_2018_real_CA'
data_train = get_data(f'{data_name}-mixed-train',
                      domain_name=f'domain/{data_name}-mixed', root_path='data_files/folktables_datasets')
domain = data_train.domain
df_train = data_train.df

data_name = f'folktables_2018_real_CA'
data_test = get_data(f'{data_name}-mixed-test',
                     domain_name=f'domain/{data_name}-mixed', root_path='data_files/folktables_datasets')
df_test = data_test.df

test_acc_original = {}
for target in domain.get_categorical_cols():
    train_acc, test_acc = linear_ml_accuracy(df_train, df_test, target, model_name=model_name)
    test_acc_original[target] = test_acc
    # print(f'target={target}: test_accuracy = {test_acc:.5f}')

sync_paths = []
acc_list = []

def run_ml_evaluation(algo:list,queries:list,epsilon_list:list,seed_list:list,model_name='LR',):
    Results = []
    for a, q in itertools.product(algo, queries):
        for T in [3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 60, 80, 25, 50, 75, 100]:
            for eps in epsilon_list:
                for seed in seed_list:
                    path = f'examples/acs/results_halfspaces/folktables_2018_real_CA/{a}/{T:03}/{eps:.2f}/sync_data_{seed}.csv'
                    if not os.path.exists(path): continue
                    print(f'reading {path}')
                    sync_paths.append(path)
                    df_train_sync = pd.read_csv(path)
                    if len(df_train_sync) > 3000:
                        df_train_sync = df_train_sync.sample(n=3000)
                    if 'Unnamed: 0' in df_train_sync.columns:
                        df_train_sync.drop('Unnamed: 0', axis=1, inplace=True)

                    cat_cols = domain.get_categorical_cols()
                    for cat in domain.get_categorical_cols():
                        df_train_sync[cat] = df_train_sync[cat].round().astype(int)

                    for target in cat_cols:
                        train_acc, test_acc = linear_ml_accuracy(df_train_sync, df_test, target, model_name=model_name)
                        acc_list.append(test_acc)

                        Results.append([f'{a}({q})', T, eps, seed, f'{target} ML Acc', test_acc_original[target], test_acc])
                        print(path, f'\ntarget ={target}: Original acc = {test_acc_original[target]:.5f},'
                                    f' Synthetic acc = {test_acc:.5f}')

    cols = ['algo', 'T', 'epsilon', 'seed', 'error type', 'original accuracy', 'private accuracy']
    df_results = pd.DataFrame(Results, columns=cols)
    df_results.to_csv(f"ML_{model_name}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ACSreal Experiment',
                                     description='Run algorithm PrivGA and RAP++ on ACSreal data')

    parser.add_argument('--algo', choices=['PrivGA', 'RAP++'], default='PrivGA')
    parser.add_argument('--queries', choices=['Halfspaces', 'Prefix', 'Ranges'], default='Prefix')
    parser.add_argument('--epsilon', type=float, default=[1], nargs='+')
    parser.add_argument('--seed', type=int, default=[0], nargs='+')
    parser.add_argument('-a', '--adaptive', action='store_true', default=True)  # on/off flag
    parser.add_argument('--rounds', type=int, default=[50], nargs='+')
    parser.add_argument('--samples_per_round', type=int, default=[10], nargs='+')

    args = parser.parse_args()
    T = [3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 60, 80, 25, 50, 75, 100]
    epsilons= [0.07, 0.23, 0.52, 0.74, 1.00]
    seeds = []
    run_ml_evaluation(algo=args.algo,queries=args.queries,)