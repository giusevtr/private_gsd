import os.path

import pandas as pd
import numpy as np





def read_tabddpm_data(dataset_name, seed, root_dir ='../tabddpm_sync_data'):

    X_num = np.load(f'{root_dir}/{dataset_name}/{seed}/X_num.npy').astype(float)
    X_cat = np.load(f'{root_dir}/{dataset_name}/{seed}/X_cat.npy', allow_pickle=True)
    y = np.load(f'{root_dir}/{dataset_name}/{seed}/y.npy')
    cat_cols = [f'cat_{i}' for i in range(X_cat.shape[1])]
    num_cols = [f'num_{i}' for i in range(X_num.shape[1])]
    all_cols = cat_cols + num_cols + ['Label']

    data_df = pd.DataFrame(np.column_stack((X_cat, X_num, y)), columns=all_cols)
    return data_df


def is_ordinal(col_df):
    vals = col_df.astype(float).values
    vals_int = col_df.astype(float).astype(int).values
    error = np.abs(vals-vals_int).max()
    return error <= 1e-9


def read_original_data(dataset_name, root_dir ='../../dp-data-dev/data2/data'):

    train_data_list = []
    val_data_list = []
    test_data_list = []

    cat_cols = []
    num_cols = []
    real_cols = []
    ordi_cols = []
    if os.path.exists(f'{root_dir}/{dataset_name}/X_cat_train.npy'):
        X_cat_train = np.load(f'{root_dir}/{dataset_name}/X_cat_train.npy')
        X_cat_val = np.load(f'{root_dir}/{dataset_name}/X_cat_val.npy')
        X_cat_test = np.load(f'{root_dir}/{dataset_name}/X_cat_test.npy')

        train_data_list.append(X_cat_train)
        val_data_list.append(X_cat_val)
        test_data_list.append(X_cat_test)

        cat_cols = [f'cat_{i}' for i in range(X_cat_train.shape[1])]

    if os.path.exists(f'{root_dir}/{dataset_name}/X_num_train.npy'):
        X_num_train = np.load(f'{root_dir}/{dataset_name}/X_num_train.npy').astype(float)
        X_num_val = np.load(f'{root_dir}/{dataset_name}/X_num_val.npy').astype(float)
        X_num_test = np.load(f'{root_dir}/{dataset_name}/X_num_test.npy').astype(float)

        train_data_list.append(X_num_train)
        val_data_list.append(X_num_val)
        test_data_list.append(X_num_test)
        num_cols = [f'num_{i}' for i in range(X_num_train.shape[1])]

    y_train = np.load(f'{root_dir}/{dataset_name}/y_train.npy')
    y_val = np.load(f'{root_dir}/{dataset_name}/y_val.npy')
    y_test = np.load(f'{root_dir}/{dataset_name}/y_test.npy')

    train_data_list.append(y_train)
    val_data_list.append(y_val)
    test_data_list.append(y_test)
    all_cols = cat_cols + num_cols + ['Label']

    train_df = pd.DataFrame(np.column_stack(train_data_list), columns=all_cols)
    val_df = pd.DataFrame(np.column_stack(val_data_list), columns=all_cols)

    train_final_df = pd.concat((train_df, val_df))

    test_df = pd.DataFrame(np.column_stack(test_data_list), columns=all_cols)

    all_df = pd.concat((train_final_df, test_df))

    for ncol in num_cols:
        if is_ordinal(all_df[ncol]):
            ordi_cols.append(ncol)
        else:
            real_cols.append(ncol)

    return train_final_df, test_df, all_df, cat_cols, ordi_cols, real_cols

