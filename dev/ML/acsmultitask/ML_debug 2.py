import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import jax.numpy as jnp
from dp_data import load_domain_config, load_df, DataPreprocessor, ml_eval
from utils import timer, Dataset, Domain, get_Xy, separate_cat_and_num_cols
from utils.cdp2adp import cdp_rho, cdp_eps
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, average_precision_score, accuracy_score


def test_ml(pca, X_train, y_train, X_test, y_test,  col_names, title=''):
    X_train_proj = pca.transform(X_train)
    X_test_proj = pca.transform(X_test)
    scorer = make_scorer(f1_score, average='macro')
    res = []
    tries = 3
    score_sum = []
    clf = LogisticRegression(solver='liblinear', penalty='l1', random_state=0)
    # clf = KNeighborsClassifier()
    clf.fit(X_train_proj, y_train)
    metric_test = scorer(clf, X_test_proj, y_test)

    # df = pd.DataFrame(clf.coef_, columns=col_names)
    df = pd.DataFrame(clf.coef_)
    df['Kind'] = title


    score_sum.append(metric_test)
    score_sum = jnp.array(score_sum)
    # print(f'LR Avg Score={score_sum.mean():.5f}\t std={score_sum.std():.5f}', end='\t\t')

    return score_sum.mean(), df
    # print()
    # return pd.DataFrame(res, columns=['Eval Data', 'Model', 'Metric', 'Score'])



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

if __name__ == "__main__":

    rescale = True
    n_components = 10

    dataset_name = 'folktables_2018_multitask_CA'
    root_path = '../../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)
    df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
    df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')

    preprocesor: DataPreprocessor
    preprocesor = pickle.load(open(f'{root_path}/{dataset_name}/preprocessor.pkl', 'rb'))

    domain = Domain.fromdict(config)
    data = Dataset(df_train, domain)
    cat_cols = domain.get_categorical_cols()
    num_cols = domain.get_numeric_cols()
    targets = ['JWMNP_bin', 'PINCP', 'MIG', 'PUBCOV', 'ESR']
    features = []
    for f in domain.attrs:
        if f not in targets:
            features.append(f)

    # for epsilon in [0.07, 0.23, 0.52, 0.74, 1]:
    for epsilon in [1.0]:
        print(f'Epsilon= {epsilon}')
        sync_df = pd.read_csv(f'sync_data/GSD/folktables_2018_multitask_CA/GSD/Binary_Tree_Marginals/{epsilon:.2f}/sync_data_0.csv')
        numeric_cols = domain.get_numeric_cols()
        for ncol in numeric_cols:
            df_real_col = df_train[[ncol]].sample(n=2000) + 0.0001
            df_real_col['Type'] = 'Real'
            df_sync_col = sync_df[[ncol]] + 0.0001
            df_sync_col['Type'] = 'Sync'

            n_mean = df_sync_col[ncol].mean()
            n_std = df_sync_col[ncol].std()

            df = pd.concat([df_real_col, df_sync_col], ignore_index=False)
            df[ncol] = (df[ncol] - n_mean) / n_std
            g = sns.histplot(data=df, x=ncol, hue='Type',
                             # log_scale=True
                             )
            g.set_yscale('log')
            plt.show()

        columns = []
        cols_num, cols_cat = separate_cat_and_num_cols(domain, features)
        for cat in cols_cat:
            for i in range(domain.size(cat)):
                columns.append(f'{cat}({i})')
        for num_c in cols_cat:
            columns.append(f'{num_c}')

        sync_df = pd.read_csv(f'sync_data/GSD/folktables_2018_multitask_CA/GSD/Binary_Tree_Marginals/{epsilon:.2f}/sync_data_0.csv')
        _, _, X_test, y_test = get_Xy(domain, features=features, target='PINCP',
                                      df_train=sync_df,
                                      df_test=df_test,
                                      rescale=rescale)

        X_train_sync, y_train_sync, X_train, y_train = get_Xy(domain, features=features, target='PINCP',
                                                  df_train=sync_df,
                                                  df_test=df_train,
                                                  rescale=rescale)

        for n_components in [10]:
            print(f'c_component = {n_components}:')

            pca_sync = PCA(n_components=n_components)
            pca_sync.fit(X_train_sync)

            ##  ML Test
            f1, real_coef_df = test_ml(pca_sync, X_train, y_train, X_test, y_test, columns, title='Real w/ Sync-PCA:')
            print(f'Real w/ Sync-PCA: f1={f1:.3f}')
            print(real_coef_df)
            f1, sync_coef_df = test_ml(pca_sync, X_train_sync, y_train_sync, X_test, y_test, columns, title='Sync w/ Sync-PCA:')
            print(f'Sync w/ Sync-PCA: f1={f1:.3f}')
            print(sync_coef_df)


            df = pd.concat([real_coef_df, sync_coef_df])

            print(df)