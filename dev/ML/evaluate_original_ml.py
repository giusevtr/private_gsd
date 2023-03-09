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
    clipped = True
    scale_real_valued = True
    dataset_name = 'folktables_2018_multitask_NY'
    root_path = '../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)

    df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
    df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')


    models = [('LR', lambda :LogisticRegression(max_iter=5000, solver='sag', random_state=0)),
              # ('RF', lambda : RandomForestClassifier( random_state=0))
              ]
    # Preprocess data.
    # df_train, df_test = filter_outliers(df_train, df_test, config, quantile=0.03, visualize_columns=False)

    domain = Domain.fromdict(config)
    features = []
    targets = ['PINCP',  'PUBCOV', 'ESR']
    for f in domain.attrs:
        if f not in targets:
            features.append(f)


    Res = []
    # for target, clipped, scale  in itertools.product(['PINCP', 'PUBCOV'], [True, False], [True, False]):
    for target, clipped, scale  in itertools.product(['PINCP'], [True, False], [True, False]):
        df_train_temp = df_train.copy()
        df_test_temp = df_test.copy()
        if clipped:
            dataset_name_temp = dataset_name + '_clipped'
            df_train_temp, df_test_temp = filter_outliers(df_train_temp, df_test_temp, config, quantile=0.01)
        else:
            dataset_name_temp = dataset_name


        X_train_orig, y_train_orig, X_test_orig, y_test_orig = get_Xy(domain, features=features, target=target,
                                                df_train=df_train_temp, df_test=df_test_temp, rescale=scale)

        #############################
        ##### Train non-private model
        #############################
        for model_name, model in models:
            clf = model()
            clf.fit(X_train_orig, y_train_orig)
            y_pred = clf.predict(X_test_orig)
            rep = classification_report(y_test_orig, y_pred, output_dict=True)
            f1 = rep['macro avg']['f1-score']
            acc = rep['accuracy']
            method = model_name
            print(f'{dataset_name_temp}, scale={scale}, {method}, target={target},  Non-Private, f1={f1}')
