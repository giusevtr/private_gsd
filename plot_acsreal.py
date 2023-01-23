import pandas as pd
from generate_plots.plot_utils_results import read_result, show_result
dataname = 'folktables_2018_mobility_CA'

data_dir = '../results'
##################################################
df = read_result(f'acsreal_results.csv', error_type='l1 error')

# df = df.loc[df['data'] == dataname, :]

show_result(df, error_type='l1 error')