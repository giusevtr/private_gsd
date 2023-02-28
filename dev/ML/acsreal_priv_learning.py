import itertools
import os.path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dp_data import load_domain_config, load_df, get_evaluate_ml, get_Xy
from ml_utils import timer, Dataset, Domain
import numpy as np


from diffprivlib.models import  LogisticRegression  as PrivLogisticRegression
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from ml_utils import filter_outliers

if __name__ == "__main__":
    # epsilon_vals = [0.07, 0.1, 0.15, 0.23, 0.52 ,0.74, 1, 2, 5, 10]
    epsilon_vals = [0.07, 0.15, 0.52,0.74, 1, 5, 10]

    # dataset_name = 'folktables_2018_real_NY'
    dataset_name = 'folktables_2018_multitask_NY'
    root_path = '../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)

    df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
    df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')
    targets = ['PINCP',  'PUBCOV', 'ESR']


    # Preprocess data.
    # df_train, df_test = filter_outliers(df_train, df_test, config, quantile=0.03, visualize_columns=False)
    scale_real_valued = True

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
    for target in ['PINCP', 'PUBCOV']:

        X_train, y_train, X_test, y_test = get_Xy(domain, features=features, target=target, df_train=df_train,
                                                  df_test=df_test, scale_real_valued=scale_real_valued)


        for seed in [0]:
            clf = LogisticRegression(max_iter=5000, random_state=seed)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print(f'Target {target}, seed={seed}')
            orig_report = classification_report(y_test, y_pred, output_dict=True)
            # print(classification_report(y_test, y_pred))
            orig_f1 = orig_report['macro avg']['f1-score']
            orig_acc = orig_report['accuracy']

            for eps in epsilon_vals:
                Res.append(['NP-LogReg', target, eps, 'F1', seed, orig_f1])
                Res.append(['NP-LogReg', target, eps, 'Accuracy', seed,  orig_acc])


                clf = PrivLogisticRegression(epsilon=eps, data_norm=data_norm, max_iter=5000, C=1)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                rep = classification_report(y_test, y_pred, output_dict=True)
                f1 = rep['macro avg']['f1-score']
                acc = rep['accuracy']
                print(f'target={target}, eps={eps}, f1={f1}')
                Res.append(['DP-LogReg', target, eps, 'F1', seed, f1])
                Res.append(['DP-LogReg', target, eps, 'Accuracy', seed, acc])


    results = pd.DataFrame(Res, columns=['Method', 'Target', 'Epsilon', 'Metric', 'Seed', 'Score'])

    sns.relplot(data=results, x='Epsilon', y='Score', col='Target', hue='Method', row='Metric', kind='line', facet_kws={'sharey':False})

    plt.show()

    print(results)
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
