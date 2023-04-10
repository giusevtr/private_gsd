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
from utils import timer, Dataset, Domain, get_Xy
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



def test_ml(pca, X_train, y_train, X_test, y_test):
    X_train_proj = pca.transform(X_train)
    X_test_proj = pca.transform(X_test)
    scorer = make_scorer(f1_score, average='macro')
    res = []
    tries = 3
    score_sum = []
    for seed in range(tries):
        clf = LogisticRegression(solver='liblinear', penalty='l1', random_state=seed)
        clf.fit(X_train_proj, y_train)
        metric_test = scorer(clf, X_test_proj, y_test)
        score_sum.append(metric_test)
    score_sum = jnp.array(score_sum)
    print(f'LR Avg Score={score_sum.mean():.5f}\t std={score_sum.std():.5f}', end='\t\t')


    # rf_score_sum = []
    # for seed in range(1):
    #     clf = GradientBoostingClassifier(random_state=seed)
    #     clf.fit(X_train_proj, y_train)
    #     metric_test = scorer(clf, X_test_proj, y_test)
    #     rf_score_sum.append(metric_test)
    # rf_score_sum = jnp.array(rf_score_sum)
    # print(f'RF Avg Score={rf_score_sum.mean():.5f}\t std={rf_score_sum.std():.5f}')

    print()
    return pd.DataFrame(res, columns=['Eval Data', 'Model', 'Metric', 'Score'])


if __name__ == "__main__":

    rescale = True
    n_components=10


    dataset_name = 'folktables_2018_multitask_CA'
    root_path = '../../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)
    # df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train').sample(n=2000, random_state=0)
    # df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test').sample(n=2000, random_state=0)

    df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
    df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')

    N = len(df_train)
    preprocesor: DataPreprocessor
    preprocesor = pickle.load(open(f'{root_path}/{dataset_name}/preprocessor.pkl', 'rb'))

    domain = Domain.fromdict(config)
    data = Dataset(df_train, domain)
    cat_cols = domain.get_categorical_cols()
    num_cols = domain.get_numeric_cols()

    print(f'train size: {df_train.shape}')
    print(f'test size:  {df_test.shape}')
    # domain = Domain.fromdict(config)
    # data = Dataset(df_train, domain)
    targets = ['JWMNP_bin', 'PINCP', 'MIG', 'PUBCOV', 'ESR']
    # targets = ['PINCP', 'PUBCOV']
    features = []
    for f in domain.attrs:
        if f not in targets:
            features.append(f)

    model = 'RandomForest'
    ml_fn = ml_eval.get_evaluate_ml(df_test, config, targets, models=[model])

    # X_train, y_train, X_test, y_test = get_Xy(domain, features=features, target='PINCP',
    #                                           df_train=df_train,
    #                                           df_test=df_test,
    #                                           rescale=rescale)
    #


    # epsilon = 0.07
    epsilon = 1
    sync_df = pd.read_csv(f'sync_data/GSD/folktables_2018_multitask_CA/GSD/Binary_Tree_Marginals/{epsilon:.2f}/sync_data_0.csv')

    N_sync = len(sync_df)
    X_train_sync, y_train_sync, X_train, y_train = get_Xy(domain, features=features, target='PINCP',
                                              df_train=sync_df,
                                              df_test=df_train,
                                              rescale=rescale)
    pca_sync = PCA(n_components=n_components)
    pca_sync.fit(X_train_sync)


    pca_real = PCA(n_components=n_components)
    pca_real.fit(X_train)

    # Visualize components
    proj_real = pca_sync.transform(X_train)
    cols = [f'x_{i}' for i in range(n_components)] + ['L']
    df_real = pd.DataFrame(np.column_stack((proj_real, y_train)), columns=cols)
    df_real['Type'] = 'Real'

    proj_sync = pca_sync.transform(X_train_sync)
    df_sync = pd.DataFrame(np.column_stack((proj_sync, y_train_sync)), columns=cols)
    df_sync['Type'] = 'Sync'
    df = pd.concat([df_real.sample(n=2000), df_sync], ignore_index=True)
    alpha = 0.1
    df1 = df[df['L'] == 1].drop(columns=['L'])
    df0 = df[df['L'] == 0].drop(columns=['L'])


    # g = sns.PairGrid(df1, hue='Type')
    # g.map_diag(sns.histplot, alpha=alpha)
    # g.map_offdiag(sns.scatterplot, alpha=alpha)
    # plt.show()
    # g = sns.PairGrid(df0, hue='Type')
    # g.map_diag(sns.histplot, alpha=alpha)
    # g.map_offdiag(sns.scatterplot, alpha=alpha)
    # plt.show()




    range_w = 10
    bins = 64
    max_error_total = 0
    avg_error_total = 0
    df = pd.concat([df_real, df_sync], ignore_index=True)
    for L in [0, 1]:
        df_L = df[df['L'] == L].drop(columns=['L'])
        for i, j in itertools.product(range(n_components), range(n_components)):
            # if i != 3: continue
            df1_real = df_L[df_L['Type'] == 'Real']
            df1_sync = df_L[df_L['Type'] == 'Sync']
            x_r = df1_real[f'x_{i}'].values
            y_r = df1_real[f'x_{j}'].values


            x_s = df1_sync[f'x_{i}'].values
            y_s = df1_sync[f'x_{j}'].values

            if i == j:
                H_real, _ = np.histogram(x_r, range=[-range_w, range_w], bins=bins)
                H_sync, _ = np.histogram(x_s, range=[-range_w, range_w], bins=bins)
                H_real = H_real.astype(float) / N
                H_sync = H_sync.astype(float) / N_sync
                errors = np.abs(H_real - H_sync)
                # plt.bar(x=np.arange(errors.shape[0]), height=errors)
                # plt.show()
                max_errror = errors.max()
                avg_errror = errors.sum()
                # print(f'Components ({i}, {j}) max error = {max_errror:.3f}, average = {avg_errror:.5f}')
            else:
                H_real, _, _ = np.histogram2d(x_r, y_r, range=[[-range_w, range_w], [-range_w, range_w]], bins=bins)
                H_sync, _, _ = np.histogram2d(x_s, y_s, range=[[-range_w, range_w], [-range_w, range_w]], bins=bins)
                H_real /= N
                H_sync /= N_sync

                errors = np.abs(H_real - H_sync)
                max_errror = errors.max()
                avg_errror = errors.sum()
                # plt.title(f'Components ({i}, {j}) max error = {max_errror:.3f}, average = {avg_errror:.5f}')
                # print(f'Components ({i}, {j}) max error = {max_errror:.3f}, average = {avg_errror:.5f}')
                # ax = sns.heatmap(errors, linewidth=0.5)
                # plt.show()

            max_error_total = max(max_error_total, max_errror)
            avg_error_total += avg_errror
    print(f'epsilon={epsilon:.2f}:\tMAX ERROR = {max_error_total:.3f}, avg_error_total={avg_error_total:.5f}')


