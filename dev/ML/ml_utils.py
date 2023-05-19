import jax.random
import matplotlib.pyplot as plt
import pandas as pd

from models import PrivGA
from stats import ChainedStatistics, Halfspace, Marginals
# from utils.utils_data import get_data
import jax.numpy as jnp
# from dp_data.data import get_data
# from dp_data import load_domain_config, load_df, get_evaluate_ml
from utils import timer, Dataset, Domain
from utils.cdp2adp import cdp_rho, cdp_eps
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, average_precision_score, accuracy_score
import numpy as np
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    LabelBinarizer,
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

def filter_outliers(df_train, df_test, config, quantile=0.01, visualize_columns=False):
    domain = Domain.fromdict(config)

    df = df_train.append(df_test)

    for num_col in domain.get_numeric_cols():
        q_lo = df[num_col].quantile(quantile)
        q_hi = df[num_col].quantile(1-quantile)

        df_train.loc[df_train[num_col] > q_hi, num_col] = q_hi
        df_test.loc[df_test[num_col] > q_hi, num_col] = q_hi

        df_train.loc[df_train[num_col] < q_lo, num_col] = q_lo
        df_test.loc[df_test[num_col] < q_lo, num_col] = q_lo

    # Rescale data
    df_outlier = df_train.append(df_test)

    for num_col in domain.get_numeric_cols():
        maxv = df_outlier[num_col].max()
        minv = df_outlier[num_col].min()

        df_train[num_col] = (df_train[num_col] - minv) / (maxv - minv)
        df_test[num_col] = (df_test[num_col] - minv) / (maxv - minv)
        # meanv = df_train[num_col].mean()
        # print(f'Col={num_col:<10}: mean={meanv:<5.3f}, min={minv:<5.3f},  max={maxv:<5.3f},')
        if visualize_columns:
            df_train[num_col].hist()
            plt.title(f'Column={num_col}')
            # plt.yscale('log')
            plt.show()

    return df_train, df_test


def separate_cat_and_num_cols(domain, features):
    # train_cols = [c for c in domain.attrs if c != target]
    train_cols_num = [c for c in features if domain[c] == 1]
    train_cols_cat = [c for c in features if c not in train_cols_num]
    return train_cols_num, train_cols_cat
def get_Xy(domain: Domain, features: list, target, df_train: pd.DataFrame, df_test: pd.DataFrame,
           scale_real_valued=True):
    cols_num, cols_cat = separate_cat_and_num_cols(domain, features)
    y_train = df_train[target].values
    y_test = df_test[target].values
    X_train = None
    X_test = None

    if len(cols_cat) > 0:
        X_cat_train = df_train[cols_cat].values
        X_cat_test = df_test[cols_cat].values
        categories = [np.arange(domain[c]) for c in cols_cat]
        enc = OneHotEncoder(categories=categories)
        enc.fit(X_cat_train)
        X_cat_train = enc.transform(X_cat_train).toarray()
        X_cat_test = enc.transform(X_cat_test).toarray()
        X_train = X_cat_train
        X_test = X_cat_test

    if len(cols_num) > 0:
        X_num_train = df_train[cols_num].values
        X_num_test = df_test[cols_num].values
        if scale_real_valued:
            scaler = StandardScaler()
            scaler.fit(X_num_train)
            X_num_train = scaler.transform(X_num_train)
            X_num_test = scaler.transform(X_num_test)

        if X_train is not None:
            X_train = np.concatenate((X_train, X_num_train), axis=1)
            X_test = np.concatenate((X_test, X_num_test), axis=1)
        else:
            X_train = X_num_train
            X_test = X_num_test

    assert X_train is not None
    assert X_test is not None
    return X_train, y_train, X_test, y_test


def evaluate_machine_learning_task(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        feature_columns: list,
        label_column: str,
        cat_columns: list,
        num_columns: list,
        endmodel=None,
        scale_real_valued=False
):
    """
    Originally by Shuai Tang and modified by Giuseppe Vietri to work for any mixed-type dataset.

    @param train_data:
    @param test_data:
    @param label_column:
    @param cat_columns:
    @param cont_columns:
    @return:
    """
    gridsearch_params = [
        (max_depth, subsample)
        for max_depth in range(5, 12)
        # for min_child_weight in range(5,8)
        for subsample in np.arange(0.2, 1.0, 0.2)
    ]
    for col in cat_columns:
        train_data[col] = train_data[col].fillna(0)
        test_data[col] = test_data[col].fillna(0)

    combined_data = pd.concat([train_data, test_data], ignore_index=True)
    #
    assert label_column not in feature_columns

    X_train = train_data.copy()
    X_test = test_data.copy()
    y_train = train_data[[label_column]]
    y_test = test_data[[label_column]]
    X_train = X_train[feature_columns]
    X_test = X_test[feature_columns]

    # X_train.drop(columns=[label_column], inplace=True)
    # X_test.drop(columns=[label_column], inplace=True)
    stored_binarizers = []
    # print('debug')
    for col in cat_columns:
        lb = LabelBinarizer()
        lb_fitted = lb.fit(combined_data[col].astype(int))
        stored_binarizers.append(lb_fitted)

    def replace_with_binarized_logit(dataframe, column_names, stored_binarizers):
        newDf = dataframe.copy()
        for idx, column_name in enumerate(column_names):
            if column_name not in newDf.columns:
                continue
            lb = stored_binarizers[idx]
            lb_results = lb.transform(newDf[column_name])
            if len(lb.classes_) <= 1:
                print(f"replace_with_binarized_legit: Error with label {column_name}")
                continue
            # columns = lb.classes_ if len(lb.classes_) > 2 else [f"is {lb.classes_[1]}"]
            columns = lb.classes_
            binarized_cols = pd.DataFrame(lb_results, columns=columns)

            newDf.drop(columns=column_name, inplace=True)
            binarized_cols.index = newDf.index
            #
            newDf = pd.concat([newDf, binarized_cols], axis=1)
        return newDf

    x_cat_cols = []
    for cat in cat_columns:
        if cat in X_train.columns:
            x_cat_cols.append(cat)

    X_train = replace_with_binarized_logit(X_train, x_cat_cols, stored_binarizers)
    X_test = replace_with_binarized_logit(X_test, x_cat_cols, stored_binarizers)


    for num_col in num_columns:
        X_num_train = X_train[num_col].values.reshape((-1, 1))
        X_num_test = X_test[num_col].values.reshape((-1, 1))
        if scale_real_valued:
            scaler = StandardScaler()
            scaler.fit(X_num_train)
            X_train[num_col] = scaler.transform(X_num_train)
            X_test[num_col] = scaler.transform(X_num_test)

    # endmodel = xgboost.XGBClassifier()
    X_train = np.array(X_train)
    y_train = np.array(y_train).ravel()
    endmodel = endmodel.fit(X_train, y_train)
    y_predict = endmodel.predict(np.array(X_test))
    # print(classification_report(y_test, y_predict))
    results = classification_report(y_test, y_predict, output_dict=True)
    return results
