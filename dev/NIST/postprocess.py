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
from dev.NIST.consistency import get_consistency_fn
from dev.NIST.consistency_simple import get_nist_simple_consistency_fn

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


NULL_COLS = [
            "MSP",
            "NOC",
            "NPF",
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
            "PWGTP",
            "WGTP"
]
from dev.NIST.consistency import INDP_CAT, INDP_CODES



def count_violations(df_orig, df, domain, preprocessor):
    data = Dataset(df_orig, domain)
    sync_data = Dataset(df, domain)
    violations_fn = get_consistency_fn(domain, preprocessor)
    real_violation_counts = violations_fn(data.to_numpy()) * len(data.df)
    print(f'\n\n\nREAL VIOLATIONS:')
    print(real_violation_counts)
    sync_violation_counts = violations_fn(sync_data.to_numpy()) * len(sync_data.df)
    print(sync_violation_counts)

def count_nulls(df_orig, df, domain):
    data = Dataset(df_orig, domain)
    sync_data = Dataset(df, domain)
    null_counter = NullCounts(domain=domain)
    null_fn = null_counter._get_dataset_statistics_fn()

    real_nulls = null_fn(data)
    sync_nulls = null_fn(sync_data)

    print(f'\n\n\nREAL NULLS:')
    print(real_nulls)
    print(f'\n\n\nSYNC NULLS:')
    print(sync_nulls)

def post_nist(df, dataset_name='national2019', nist_type='simple'):

    # dataset_name = 'national2019'
    root_path = '../../dp-data-dev/datasets/preprocessed/sdnist_dce/'
    config = load_domain_config(dataset_name, root_path=root_path)
    df_orig = load_df(dataset_name, root_path=root_path)
    preprocessor_path = os.path.join(root_path +dataset_name, 'preprocessor.pkl')
    with open(preprocessor_path, 'rb') as handle:
        preprocessor = pickle.load(handle)

    def get_encoded_value(feature, value):
        if feature in preprocessor.attrs_cat:
            enc = preprocessor.encoders[feature]
            value = str(value)
            v = pd.DataFrame(np.array([value]), columns=[feature])
            return enc.transform(v)[0]
        if feature in preprocessor.mappings_ord.keys():
            min_val, _ = preprocessor.mappings_ord[feature]
            return value - min_val
    orig_domain = Domain(config)


    all_cols = orig_domain.attrs
    if nist_type == 'simple':
        all_cols.remove('INDP')
        all_cols.remove('WGTP')
        all_cols.remove('PWGTP')

        del preprocessor.mappings_cat['INDP']
        del preprocessor.mappings_num['WGTP']
        del preprocessor.mappings_num['PWGTP']

        # consistency_fn = get_nist_simple_population_consistency_fn(domain, preprocessor)
    elif nist_type == 'all':
        all_cols.remove('INDP_CAT')
        # consistency_fn = get_nist_all_population_consistency_fn(domain, preprocessor)
    domain = orig_domain.project(all_cols)

    preprocessor.attrs_cat = domain.get_categorical_cols()
    preprocessor.attrs_num = domain.get_numerical_cols()
    preprocessor.attrs_ord = domain.get_ordinal_cols()

    INDP_MAP = {}

    for indp, indp_cat in INDP_CODES.items():
        if indp == 'N': continue
        try:
            if nist_type == 'all':
                indp_enc = get_encoded_value('INDP', f'{int(indp):>03}')
                indp_cat_enc = get_encoded_value('INDP_CAT', INDP_CAT[indp_cat])
                INDP_MAP[indp_enc] = indp_cat_enc
        except:
            pass

    def map_indp(cat):
        if np.isnan(cat): return np.nan
        indp_cat = INDP_MAP[cat]
        return indp_cat
    if nist_type == 'all':
        df['INDP_CAT'] = df['INDP'].apply(map_indp)

    real_data_df = df_orig[df.columns]

    ## COUNT NULL VALUES
    count_nulls(real_data_df, df, domain)

    # Remove inconsistencies
    sync_data = Dataset(df, domain)
    con_fn = None
    if nist_type == 'all':
        con_fn = get_consistency_fn(domain, preprocessor, axis=1)
    elif nist_type == 'simple':
        con_fn = get_nist_simple_consistency_fn(domain, preprocessor, axis=1)
    row_consistency = con_fn(sync_data.to_numpy())
    rem_rows_idx = row_consistency>0
    drop = np.argwhere(rem_rows_idx).flatten()
    drop_index = df.iloc[drop].index
    df = df.drop(index=drop_index)
    print(f'Removing ', rem_rows_idx.sum(), ' inconsistent rows')

    temp_cat_cols = ['RAC1P', 'DEAR', 'SEX', 'PUMA', 'DEYE', 'HOUSING_TYPE']
    df[temp_cat_cols] = df[temp_cat_cols].fillna(0).astype(int)
    df["DENSITY"] = df["DENSITY"].fillna(0).astype(float)

    df_post = preprocessor.inverse_transform(df)

    REAL = domain.get_numerical_cols()
    for col in REAL:
        df_post[col] = df_post[col].astype(str)
        df_post[col] = df_post[col].replace(to_replace='nan', value='N')

    INTS = ["PUMA", "SEX", "HISP", "MSP", "RAC1P", "HOUSING_TYPE", "OWN_RENT", "INDP",
                     "INDP_CAT", "DREM", "DPHY", "DEYE", "DEAR"] + ['AGEP', 'POVPIP'] +\
           ["NOC", "NPF", "EDU", "PINCP_DECILE", "DVET"]
    if nist_type == 'simple':
        INTS.remove('INDP')
    for col in INTS:
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

    # df_post = df_post[ALL_COLS]

    NON_NULL = [
        "AGEP",
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
    if nist_type == 'simple':
        NON_NULL.remove('PWGTP')
        NON_NULL.remove('WGTP')
    for int_feat in NON_NULL:
        int_col = df_post[int_feat]

        new_col = int_col.replace('N', 0)
        new_col2 = new_col.astype(float).round().astype(int)
        df_post[int_feat] = new_col2.values

    # Manual fix: These integer features are not allowed to have null values. Remove all null values.
    # df_post.loc[ints].replace('N', 0, inplace=True)
    # df_post = df_post.astype(float).round().astype(int)

    copies = len(df_orig) // len(df_post)
    df_post_upsample = pd.concat([df_post.copy() for _ in range(copies)])
    rem = len(df_orig) - len(df_post_upsample)
    df_post_upsample = pd.concat([df_post_upsample, df_post.sample(n=rem)])
    return df_post_upsample

target_dataset = sys.argv[1]  # options: national2019, tx2019, ma2019
assert target_dataset in ['national2019', 'tx2019', 'ma2019']
nist_type = sys.argv[2]  # options: simple, all
assert nist_type in ['simple', 'all']
sync_path = sys.argv[3]
save_post_pat = sys.argv[4]

print('Reading', sync_path)
df = pd.read_csv(sync_path)
df_post = post_nist(df, dataset_name=target_dataset, nist_type=nist_type)
print('Saving',save_post_pat)
df_post.to_csv( save_post_pat, index=False)


