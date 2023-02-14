# RAC1P

from utils.utils_data import get_data
import pandas as pd
import numpy as np
task = 'coverage'
data_name = f'folktables_2018_{task}_CA'
data = get_data(f'{data_name}-mixed-train', domain_name=f'domain/{data_name}-cat',
                root_path='../../../data_files/folktables_datasets')


df_real = data.df
races = df_real['RAC1P'].value_counts()
print(races)

df_minority = df_real[df_real['RAC1P'] == 6]

print(len(df_minority))


def query(df_data, att1, value1, att_given, value_given):
    df_given = df_data[(df_data[att_given] == value_given)]

    df1 = df_given[df_given[att1] == value1]

    count_numerator = len(df1)
    count_denominator = len(df_given)
    return count_numerator / count_denominator


df_sync = pd.read_csv('2way_only/folktables_2018_coverage_CA/1.0/sync_data_0.csv')

query_value = 1
query_answer_real = query(df_real, 'PUBCOV', query_value, 'RAC1P', 6)
query_answer_sync = query(df_sync, 'PUBCOV', query_value, 'RAC1P', 6)
print('minority query:', query_answer_real, query_answer_sync, np.abs(query_answer_real - query_answer_sync))

query_answer_real = query(df_real, 'PUBCOV', query_value, 'RAC1P', 0)
query_answer_sync = query(df_sync, 'PUBCOV', query_value, 'RAC1P', 0)
print('majority query:', query_answer_real, query_answer_sync, np.abs(query_answer_real - query_answer_sync))


# df[df['RAC1P'] == '']
