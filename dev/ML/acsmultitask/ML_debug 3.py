import itertools
import os.path
import pickle

import jax.random
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from sklearn.decomposition import PCA
from models import PrivGA, SimpleGAforSyncData
from stats import ChainedStatistics, Halfspace, Marginals
# from utils.utils_data import get_data
import jax.numpy as jnp
# from dp_data.data import get_data
from dp_data import load_domain_config, load_df, DataPreprocessor, ml_eval
from utils import timer, Dataset, Domain, get_Xy, separate_cat_and_num_cols
from utils.cdp2adp import cdp_rho, cdp_eps
import numpy as np
from dev.ML.ml_utils import evaluate_machine_learning_task

from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, average_precision_score, accuracy_score


def test_ml(X_train, y_train, X_test, y_test,  col_names, title=''):
    scorer = make_scorer(f1_score, average='macro')
    score_sum = []
    clf = LogisticRegression(solver='liblinear', penalty='l1', random_state=0)
    # clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    metric_test = scorer(clf, X_test, y_test)

    df = pd.DataFrame(clf.coef_, columns=col_names)
    df['Kind'] = title

    score_sum.append(metric_test)
    score_sum = jnp.array(score_sum)

    return score_sum.mean(), df



def test_rf(pca, X_train, y_train, X_test, y_test, title=''):
    X_train_proj = pca.transform(X_train)
    X_test_proj = pca.transform(X_test)

    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train_proj, y_train)
    # y_pred = clf.predict(X_test_proj)
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Plot the impurity-based feature importances of the forest
    feature_names = [f'C{i}' for i in range(X_test_proj.shape[0])]
    x_labels = [feature_names[x] for x in indices]
    plt.figure()
    plt.title(f"Feature importances {title}")
    plt.bar(x=x_labels, height=importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(rotation=45)
    plt.xlim([-1, X_train_proj.shape[1]])
    plt.show()

    scorer = make_scorer(f1_score, average='macro')
    metric_test = scorer(clf, X_test_proj, y_test)
    return metric_test


RESCALE =False

def plot_marginal_dist(real_df, sync_df, feature_col, label_col):

    real_col_df = real_df[[feature_col, label_col]]
    real_col_df['Kind'] = 'Real'
    sync_col_df = sync_df[[feature_col, label_col]]
    sync_col_df['Kind'] = 'Sync'

    if RESCALE:
        sync_mean = sync_col_df[feature_col].mean()
        sync_std = sync_col_df[feature_col].mean()

        sync_col_df[feature_col] = (sync_col_df[feature_col] - sync_mean) / sync_std
        real_col_df[feature_col] = (real_col_df[feature_col] - sync_mean) / sync_std

    def hist_plot(x, **kwargs):
        # data = pd.concat([x, y], ignore_index=True)
        # sns.histplot(data=data, x='')

        plt.hist(x, **kwargs)

    df = pd.concat([real_col_df, sync_col_df], ignore_index=True)
    df[feature_col] = df[feature_col] + 0.0001
    # g = sns.FacetGrid(data=df, row='Kind', hue=label_col)
    # g.map(hist_plot, x)
    sns.displot(
        df, x=feature_col,hue=label_col, row="Kind",
        # binwidth=3, height=3,
        log_scale=(False, True),
        facet_kws=dict(margin_titles=True),
    )
    plt.show()

if __name__ == "__main__":

    rescale = True

    dataset_name = 'folktables_2018_multitask_CA'
    root_path = '../../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)
    train_df = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train').sample(2000)
    test_df = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')

    preprocesor: DataPreprocessor
    preprocesor = pickle.load(open(f'{root_path}/{dataset_name}/preprocessor.pkl', 'rb'))

    domain = Domain.fromdict(config)
    data = Dataset(train_df, domain)
    cat_cols = domain.get_categorical_cols()
    num_cols = domain.get_numeric_cols()
    targets = ['JWMNP_bin', 'PINCP', 'MIG', 'PUBCOV', 'ESR']
    features = []
    for f in domain.attrs:
        if f not in targets:
            features.append(f)

    epsilon = 1

    columns = []
    cols_num, cols_cat = separate_cat_and_num_cols(domain, features)
    for cat in cols_cat:
        for i in range(domain.size(cat)):
            columns.append(f'{cat}({i})')
    for num_c in cols_num:
        columns.append(f'{num_c}')

    # sync_df = pd.read_csv(f'sync_data/GSD/folktables_2018_multitask_CA/GSD/Binary_Tree_Marginals/{epsilon:.2f}/sync_data_0.csv')
    sync_df = pd.read_csv(f'sync_data/GSD/folktables_2018_multitask_CA/GSD/2Cat+Prefix/1.00/sync_data_0.csv')


    plot_marginal_dist(train_df, sync_df, feature_col='WKHP', label_col='PINCP')
    plot_marginal_dist(train_df, sync_df, feature_col='INTP', label_col='PINCP')
    plot_marginal_dist(train_df, sync_df, feature_col='SEMP', label_col='PINCP')
    plot_marginal_dist(train_df, sync_df, feature_col='WAGP', label_col='PINCP')



    _, _, X_test, y_test = get_Xy(domain, features=features, target='PINCP',
                                  df_train=sync_df,
                                  df_test=test_df,
                                  rescale=rescale)

    X_train_sync, y_train_sync, X_train, y_train = get_Xy(domain, features=features, target='PINCP',
                                                          df_train=sync_df,
                                                          df_test=train_df,
                                                          rescale=rescale)


    ##  ML Test
    f1, real_coef_df = test_ml(X_train, y_train, X_test, y_test, columns, title='Real w/ Sync-PCA:')
    print(f'Real w/ Sync-PCA: f1={f1:.3f}')
    f1, sync_coef_df = test_ml(X_train_sync, y_train_sync, X_test, y_test, columns, title='Sync w/ Sync-PCA:')
    print(f'Sync w/ Sync-PCA: f1={f1:.3f}')


    df = pd.concat([real_coef_df, sync_coef_df])


    print(df)