import os
import sys

import pandas as pd
from dp_data import load_domain_config, load_df
from utils import timer, Dataset, Domain, filter_outliers
import pickle
from stats import NullCounts
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def post_nist(df):
    ALL_COLS = ["PUMA",
            "AGEP",
            "SEX",
            "MSP",
            "HISP",
            "RAC1P",
            "NOC",
            "NPF",
            "HOUSING_TYPE",
            "OWN_RENT",
            "DENSITY",
                "INDP",
                "INDP_CAT",
                "EDU",
                "PINCP",
                "PINCP_DECILE",
                "POVPIP",
                "DVET",
                "DREM",
                "DPHY",
                "DEYE",
                "DEAR",
                "PWGTP",
                "WGTP"
                ]
    dataset_name = 'national2019'
    root_path = '../../../dp-data-dev/datasets/preprocessed/sdnist_dce/'
    config = load_domain_config(dataset_name, root_path=root_path)
    df_orig = load_df(dataset_name, root_path=root_path)
    preprocessor_path = os.path.join(root_path +dataset_name, 'preprocessor.pkl')
    with open(preprocessor_path, 'rb') as handle:
        # preprocessor:
        preprocessor = pickle.load(handle)
        temp: pd.DataFrame
        # df_orig.fillna('N', inplace=True)
        # df_orig = df_orig.astype('str')
        temp = preprocessor.inverse_transform(df_orig)
        print(temp)

    domain = Domain(config)


    data = Dataset(df_orig, domain)

    # df = pd.read_csv('sync_national.csv')

    nulls_module = NullCounts(domain)
    nulls_fn = nulls_module._get_dataset_statistics_fn()
    sync_data = Dataset(df, domain)
    print(f'orig nulls count: ', nulls_fn(data))
    print(f'sync nulls count: ', nulls_fn(sync_data))



    temp_cat_cols = ['RAC1P', 'DEAR', 'SEX', 'PUMA', 'DEYE', 'HOUSING_TYPE']
    df[temp_cat_cols] = df[temp_cat_cols].fillna(0).astype(int)

    df["DENSITY"] = df["DENSITY"].fillna(0).astype(float)

    cutoff =0.2
    sync = df.sample(n=len(df_orig), replace=True)
    real = df_orig[df_orig['PINCP'] < cutoff]['PINCP'].to_frame()
    real['Type'] = 'Real'
    temp = sync[sync['PINCP'] < cutoff]['PINCP'].to_frame()
    temp['Type'] = 'Sync'
    df_income = pd.concat([real, temp])
    sns.histplot(data=df_income, x='PINCP', hue='Type', bins=100)
    plt.show()

    df_post = preprocessor.inverse_transform(df)


    REAL = domain.get_numerical_cols()
    for col in REAL:
        df_post[col] = df_post[col].astype(str)
        df_post[col] = df_post[col].replace(to_replace='nan', value='N')

    INTS = ["PUMA", "SEX", "HISP", "MSP", "RAC1P", "HOUSING_TYPE", "OWN_RENT", "INDP",
                     "INDP_CAT", "DREM", "DPHY", "DEYE", "DEAR"] + ['AGEP', 'POVPIP', 'PWGTP', 'WGTP'] +\
           ["NOC", "NPF", "EDU", "PINCP_DECILE", "DVET"]
    for col in INTS:
        print('col', col, ': type=', df_post[col].dtypes)
        if df_post[col].dtypes == 'float64':
            df_post[col] = df_post[col].round()
            df_post[col] = df_post[col].fillna(-1000000)
            df_post[col] = df_post[col].astype(int)
            df_post[col] = df_post[col].astype(str)
            df_post[col] = df_post[col].replace('-1000000', 'N')
        elif df_post[col].dtypes == 'object':
            df_post[col] = df_post[col].fillna('N')


    # Manual fix: Replace all values of 8 by null
    df_post['NOC'] = df_post['NOC'] .replace('8', 'N')

    df_post = df_post[ALL_COLS]
    # df_post["PWGTP"].replace('N', 0).astype(float).round().astype(int)
    # df_post["WGTP"].replace('N', 0).astype(float).round().astype(int)
    # df_post["AGEP"].replace('N', 0).astype(float).round().astype(int)

    ints = [
            "OWN_RENT",
            "SEX",
            "HISP",
            "RAC1P",
            "HOUSING_TYPE",
            "OWN_RENT",
            "DEYE",
            "DEAR",
            "PWGTP",
            "WGTP"
    ]
    for int_feat in ints:
        int_col = df_post[int_feat]

        new_col = int_col.replace('N', 0)
        new_col2 = new_col.astype(float).round().astype(int)
        df_post[int_feat] = new_col2.values

    # Manual fix: These integer features are not allowed to have null values. Remove all null values.
    # df_post.loc[ints].replace('N', 0, inplace=True)
    # df_post = df_post.astype(float).round().astype(int)


    df_post = df_post.sample(n=len(df_orig), replace=True)
    print(df_post)

    copies = len(df_orig)  // len(df_post)
    df_post_upsample = pd.concat([df_post.copy() for _ in range(copies)])
    rem = len(df_post) - len(df_post_upsample)
    df_post_upsample = pd.concat([df_post_upsample, df_post.sample(n=rem)])
    return df_post_upsample

sync_path = sys.argv[1]
save_post_pat = sys.argv[2]

# df = pd.read_csv('sync_data/national2019/GSD/Ranges/oneshot/10.00/sync_data_0.csv')
df = pd.read_csv(sync_path)
df_post = post_nist(df)
print(save_post_pat)
df_post.to_csv(save_post_pat, index=False)





# df = pd.read_csv('sync_data/national2019/GSD/Ranges/oneshot/10.00/sync_data_0.csv')
# df = pd.read_csv('sync_data_adaptive_10000.csv')
# df = pd.read_csv('sync_data_0_adaptive.csv')
# df = pd.read_csv('sync_national.csv')








# nulls_module = NullCounts(domain)
# nulls_fn = nulls_module._get_dataset_statistics_fn()
# sync_data = Dataset(df, domain)
# print(f'orig nulls count: ', nulls_fn(data))
# print(f'sync nulls count: ', nulls_fn(sync_data))
# temp_cat_cols = ['RAC1P', 'DEAR', 'SEX', 'PUMA', 'DEYE', 'HOUSING_TYPE']
# df[temp_cat_cols] = df[temp_cat_cols].fillna(0).astype(int)

# df_post = preprocessor.inverse_transform(df)
# REAL = domain.get_numerical_cols()
# for col in REAL:
#     df_post[col] = df_post[col].astype(str)
#     df_post[col] = df_post[col].replace(to_replace='nan', value='N')
#
# INTS = ["PUMA", "SEX", "HISP", "MSP", "RAC1P", "HOUSING_TYPE", "OWN_RENT", "INDP",
#                  "INDP_CAT", "DREM", "DPHY", "DEYE", "DEAR"] + ['AGEP', 'POVPIP', 'PWGTP', 'WGTP'] +\
#        ["NOC", "NPF", "EDU", "PINCP_DECILE", "DVET"]
# for col in INTS:
#     print('col', col, ': type=', df_post[col].dtypes)
#     if df_post[col].dtypes == 'float64':
#         df_post[col] = df_post[col].round()
#         df_post[col] = df_post[col].fillna(-1000000)
#         df_post[col] = df_post[col].astype(int)
#         df_post[col] = df_post[col].astype(str)
#         df_post[col] = df_post[col].replace('-1000000', 'N')
#     elif df_post[col].dtypes == 'object':
#         df_post[col] = df_post[col].fillna('N')



# Manual fix: Replace all values of 8 by null
# df_post['NOC'] = df_post['NOC'] .replace('8', 'N')
# df_post = df_post[ALL_COLS]
# df_post["DENSITY"] = df_post["DENSITY"].replace('N', 0).astype(float)
# ints = [
#         "AGEP",
#         "SEX",
#         "HISP",
#         "RAC1P",
#         "HOUSING_TYPE",
#         "OWN_RENT",
#         "DEYE",
#         "DEAR",
#         "PWGTP",
#         "WGTP"
# ]
# # Manual fix: These integer features are not allowed to have null values. Remove all null values.
# df_post[ints] = df_post[ints].replace('N', 0).astype(float).round().astype(int)
# df_post = df_post.sample(n=len(df_orig), replace=True)