import itertools
import os.path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dp_data import load_domain_config, load_df
from utils import timer, Dataset, Domain, get_Xy
import numpy as np


# from diffprivlib.models import  LogisticRegression  as PrivLogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from ml_utils import filter_outliers

if __name__ == "__main__":
    # epsilon_vals = [0.07, 0.1, 0.15, 0.23, 0.52 ,0.74, 1, 2, 5, 10]
    evaluate_original = False
    epsilon_vals = [0.07, 0.23, 0.52, 0.74, 1, 10]
    seeds = [0]
    Method = 'PrivGA'
    # Method = 'RAP'

    # dataset_name = 'folktables_2018_real_NY'
    dataset_name = 'folktables_2018_multitask_NY'
    root_path = '../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)

    df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
    df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')
    targets = ['PINCP',  'PUBCOV', 'ESR']


    models = [('LG', lambda :LogisticRegression(max_iter=5000, random_state=0)),
              ('RF', lambda : RandomForestClassifier(random_state=0))]
    # Preprocess data.
    # df_train, df_test = filter_outliers(df_train, df_test, config, quantile=0.03, visualize_columns=False)
    scale_real_valued = True

    domain = Domain.fromdict(config)
    features = []
    for f in domain.attrs:
        if f not in targets:
            features.append(f)

    Res = []
    for target in ['PINCP', 'PUBCOV']:
        X_train, y_train, X_test, y_test = get_Xy(domain, features=features, target=target, df_train=df_train,
                                                  df_test=df_test, rescale=scale_real_valued)

        #############################
        ##### Train non-private model
        #############################
        if evaluate_original:
            for model_name, model in models:
                clf = model()
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                rep = classification_report(y_test, y_pred, output_dict=True)
                f1 = rep['macro avg']['f1-score']
                acc = rep['accuracy']
                method = model_name
                print(f'{dataset_name}, {method}, target={target},  Non-Private, f1={f1}')
                for eps in epsilon_vals:
                    Res.append([dataset_name, 'No', method, target, eps, 'F1', 0, f1])
                    Res.append([dataset_name, 'No', method, target, eps, 'Accuracy', 0, acc])

        for eps in epsilon_vals:
            for seed in seeds:
                sync_path = ''
                if Method == 'PrivGA':
                    sync_path = f'../examples/acsmulti/sync_data/{dataset_name}/PrivGA/Ranges/oneshot/{eps:.2f}/sync_data_{seed}.csv'
                elif Method == 'RAP':
                    sync_path = f'../sync_data/{dataset_name}/RAP/Ranges/oneshot/{eps:.2f}/sync_data_{seed}.csv'

                if not os.path.exists(sync_path): continue
                df_sync = pd.read_csv(sync_path, index_col=None)

                X_train, y_train, X_test, y_test = get_Xy(domain, features=features, target=target, df_train=df_sync,
                                                          df_test=df_test, rescale=scale_real_valued)

                for model_name, model in models:
                    clf = model()
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    rep = classification_report(y_test, y_pred, output_dict=True)
                    f1 = rep['macro avg']['f1-score']
                    acc = rep['accuracy']
                    method = Method+'+'+model_name
                    print(f'{dataset_name}, {method}, target={target},  eps={eps}, f1={f1}')
                    Res.append([dataset_name, 'Yes', method, target, eps, 'F1', seed, f1])
                    Res.append([dataset_name, 'Yes', method, target, eps, 'Accuracy', seed, acc])


    results = pd.DataFrame(Res, columns=['Dataset', 'Is DP', 'Method', 'Target', 'Epsilon', 'Metric', 'Seed', 'Score'])
    print(results)
    if os.path.exists('results.csv'):
        results_pre = pd.read_csv('results.csv', index_col=None)
        results = results_pre.append(results)
    print(f'Saving results.csv')
    results.to_csv('results.csv', index=False)

    # df_sync1 = pd.read_csv('folktables_2018_multitask_NY_sync_0.07_0.csv')
    # df_sync2 = pd.read_csv('folktables_2018_multitask_NY_sync_1.00_0.csv')
    #
    # print('Synthetic train eps=0.07:')
    # results = ml_eval_fn(df_sync1, 0)
    # results = results[results['Metric'] == 'f1']
    # print(results)
    #
    # print('Synthetic train eps=1.00:')
    # results = ml_eval_fn(df_sync2, 0)
    # results = results[results['Metric'] == 'f1']
    # print(results)
