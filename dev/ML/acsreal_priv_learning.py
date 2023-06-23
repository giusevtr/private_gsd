import itertools
import os.path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils import timer, Dataset, Domain, get_Xy

import numpy as np


from diffprivlib.models import LogisticRegression as PrivLogisticRegression
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from dev.dataloading.data_functions.acs import get_acs_all
from ml_utils import filter_outliers, evaluate_machine_learning_task

if __name__ == "__main__":
    epsilon_vals = [0.07, 0.23, 0.52, 0.74, 1]
    # epsilon_vals = [0.07, 0.23, 0.52, 0.74, 1]
    seeds = [0, 1, 2]

    evaluate_original = True
    # epsilon_vals = [10]
    # seeds = [0]

    # dataset_name = 'acs_multitask_NY'
    dataset_name = 'folktables_2014_multitask_NY'
    data_all, data_container_fn = get_acs_all(state='NY')
    data_container = data_container_fn(seed=0)

    domain = data_container.train.domain
    df_train = data_container.from_dataset_to_df_fn(
        data_container.train
    )
    df_test = data_container.from_dataset_to_df_fn(
        data_container.test
    )

    cat_cols = domain.get_categorical_cols()
    num_cols = domain.get_numeric_cols()

    # dataset_name = 'folktables_2018_real_NY'
    targets = ['PINCP',  'PUBCOV', 'ESR', 'MIG', 'JWMNP']

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
    for target in targets:
    # for target in ['PUBCOV']:


        if evaluate_original:


            for seed in seeds:
                clf = LogisticRegression(max_iter=5000, random_state=seed,
                                         solver='liblinear', penalty='l1')

                ml_result = evaluate_machine_learning_task(df_train, df_test,
                                               feature_columns=features,
                                               label_column=target,
                                               cat_columns=cat_cols,
                                               num_columns=num_cols,
                                               endmodel=clf)
                method = 'LR'
                f1 = ml_result['macro avg']['f1-score']
                acc = ml_result['accuracy']
                print(f'{dataset_name}, {method}, target={target},  Non-Private, f1={f1}')
                for eps in epsilon_vals:
                    Res.append([dataset_name, 'No', 'Original', method, target, eps, 'F1', 0, f1])
                    Res.append([dataset_name, 'No', 'Original', method, target, eps, 'Accuracy', 0, acc])


        for seed in seeds:
            for eps in epsilon_vals:
                clf = PrivLogisticRegression(epsilon=eps, data_norm=data_norm, max_iter=5000, C=1)
                rep = evaluate_machine_learning_task(df_train, df_test,
                                                           feature_columns=features,
                                                           label_column=target,
                                                           cat_columns=cat_cols,
                                                           num_columns=num_cols,
                                                           endmodel=clf,
                                                     scale_real_valued=True)
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
