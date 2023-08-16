# import pandas as pd
# import os
# from models import GSD
# from utils import Dataset, Domain
# import numpy as np
# from utils import MLEncoder
# import matplotlib.pyplot as plt
# from sklearn.metrics import f1_score, make_scorer
# from xgboost import XGBClassifier
import itertools

import pandas as pd
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from experiments.run_exp import ml_evaluation, read_tabddpm_data, get_ml_score_fn, read_original_data
from utils import MLEncoder
from sklearn.metrics import f1_score, make_scorer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
score_fn = make_scorer(f1_score, average='macro')

import seaborn as sns
# QUANTILES = 50
names = [
    # "Nearest Neighbors",
    "Logistic Regression",
    # "Gaussian Process",
    # "Decision Tree",
    "Random Forest",
    # "Neural Net",
    # "AdaBoost",
    # "Naive Bayes",
    "XGboost"
]

classifiers = [
    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.025),
    LogisticRegression(max_iter=5000),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    RandomForestClassifier(),
    # AdaBoostClassifier(),
    # GaussianNB(),
    XGBClassifier()
]

def run_classifiers( X_train, y_train, X_test, y_test):
    res = []
    for name, clf in zip(names, classifiers):
        scaler = StandardScaler()
        scaler.fit(X_train)
        clf.fit(scaler.transform(X_train), y_train)
        train_score = score_fn(clf, scaler.transform(X_train), y_train)
        test_score =  score_fn(clf, scaler.transform(X_test), y_test)

        print(f'{name:<20}: train.score={train_score:.6f}\ttest.score={test_score:.6f}')
        res.append((name, train_score, test_score))
    res_df = pd.DataFrame(res, columns = ['Model', 'Train Score', 'Test Score'])
    return res_df



SEEDS = [0, 1, 2]

datasets = ['adult', 'churn2']

results = []
for data_name, seed in itertools.product(datasets, SEEDS):

    train_df, test_df, all_df, cat_cols, ord_cols, real_cols = read_original_data(data_name, root_dir='data2/data')
    encoder = MLEncoder(cat_features=cat_cols, num_features=ord_cols + real_cols,
                        target='Label', rescale=False)
    encoder.fit(all_df)
    X_train_oh, y_train = encoder.encode(train_df)
    X_test_oh, y_test = encoder.encode(test_df)

    # Original
    print('Original')
    orig_df = run_classifiers(X_train_oh, y_train, X_test_oh, y_test)

    print('GSD')
    sync_dir = f'sync_data/{data_name}/N/2/5000/0.990/inf/{seed}'
    gsd_X_train_oh, gsd_y_train = encoder.encode(pd.read_csv(f'{sync_dir}/sync_data.csv'))
    gsd_df = run_classifiers( gsd_X_train_oh, gsd_y_train, X_test_oh, y_test)

    print('TabDDPM')
    tab_df = read_tabddpm_data(dataset_name=data_name, seed=0, root_dir='tabddpm_sync_data')
    tab_X_train_oh, tab_y_train = encoder.encode(tab_df)
    tabddpm_df = run_classifiers( tab_X_train_oh, tab_y_train, X_test_oh, y_test)

    orig_df['Data'] = data_name
    orig_df['Generator'] = 'Original'
    orig_df['seed'] = seed

    gsd_df['Data'] = data_name
    gsd_df['Generator'] = 'GSD'
    gsd_df['seed'] = seed


    tabddpm_df['Data'] = data_name
    tabddpm_df['Generator'] = 'TabDDPM'
    tabddpm_df['seed'] = seed

    results.append(orig_df)
    results.append(gsd_df)
    results.append(tabddpm_df)

results_df = pd.concat(results, ignore_index=True)
results_df.to_csv('results/ml_results.csv')
print(results_df)
print(results_df.groupby(['Data', 'Generator', 'Model']).mean())




