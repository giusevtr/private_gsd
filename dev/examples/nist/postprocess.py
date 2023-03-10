import os
import pandas as pd
from dp_data import load_domain_config, load_df
from utils import timer, Dataset, Domain, filter_outliers
import pickle
from stats import NullCounts
import numpy as np

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

df = pd.read_csv('sync_data/national2019/GSD/Ranges/oneshot/10.00/sync_data_0.csv')
# df = pd.read_csv('sync_national.csv')

nulls_module = NullCounts(domain)
nulls_fn = nulls_module._get_dataset_statistics_fn()
sync_data = Dataset(df, domain)
print(f'orig nulls count: ', nulls_fn(data))
print(f'sync nulls count: ', nulls_fn(sync_data))



temp_cat_cols = ['RAC1P', 'DEAR', 'SEX', 'PUMA', 'DEYE', 'HOUSING_TYPE']
df[temp_cat_cols] = df[temp_cat_cols].fillna(0).astype(int)

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
df_post["DENSITY"] = df_post["DENSITY"].replace('N', 0).astype(float)
# df_post["AGEP"].replace('N', 0).astype(float).round().astype(int)

ints = [
        "AGEP",
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
# Manual fix: These integer features are not allowed to have null values. Remove all null values.
df_post[ints] = df_post[ints].replace('N', 0).astype(float).round().astype(int)


df_post = df_post.sample(n=len(df_orig), replace=True)
df_post.to_csv('gsd_national.csv', index=False)
print(df_post)