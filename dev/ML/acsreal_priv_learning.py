import itertools
import os.path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dp_data import load_domain_config, load_df
from utils import timer, Dataset, Domain, get_Xy

import numpy as np


from diffprivlib.models import LogisticRegression  as PrivLogisticRegression
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from ml_utils import filter_outliers

if __name__ == "__main__":
    epsilon_vals = [0.07, 0.23, 0.52, 0.74, 1, 10]
    # epsilon_vals = [0.07, 0.23, 0.52, 0.74, 1]
    seeds = [0, 1, 2]

    scale_real_valued = True
    # epsilon_vals = [10]
    # seeds = [0]


    # dataset_name = 'folktables_2018_real_NY'
    dataset_name = 'folktables_2018_multitask_NY'
    root_path = '../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)

    df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
    df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')
    targets = ['PINCP',  'PUBCOV', 'ESR']

    domain = Domain.fromdict(config)
    features = []
    for f in domain.attrs:
        if f not in targets:
            features.append(f)

    data_norm = np.sqrt(len(features))
    print(f'data_norm={data_norm}')
    # ml_eval_fn = get_evaluate_ml(df_test, config, targets=targets, models=['LogisticRegression'])
    # orig_results = ml_eval_fn(df_train.sample(n=20000), 0 )
    # print(orig_results)


    print(f'Private Logistic Regression:')
    Res = []
    # for target in ['PINCP']:
    for target in ['PUBCOV']:

        X_train, y_train, X_test, y_test = get_Xy(domain, features=features, target=target, df_train=df_train,
                                                  df_test=df_test, rescale=scale_real_valued)


        for seed in seeds:
            for eps in epsilon_vals:

                clf = PrivLogisticRegression(epsilon=eps, data_norm=data_norm, max_iter=5000, C=1)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                rep = classification_report(y_test, y_pred, output_dict=True)
                f1 = rep['macro avg']['f1-score']
                acc = rep['accuracy']
                print(f'{dataset_name}, target={target}, eps={eps}, f1={f1}')
                Res.append([dataset_name, 'Yes', 'DP-Obj-Per', 'LR', target, eps, 'F1', seed, f1])
                Res.append([dataset_name, 'Yes', 'DP-Obj-Per', 'LR', target, eps, 'Accuracy', seed, acc])


    results = pd.DataFrame(Res, columns=['Dataset', 'Is DP', 'Method', 'Model', 'Target', 'Epsilon', 'Metric', 'Seed', 'Score'])

    if os.path.exists('results.csv'):
        results_pre = pd.read_csv('results.csv', index_col=None)
        results = results_pre.append(results)
    print(f'Saving results.csv')
    results.to_csv('results.csv', index=False)
