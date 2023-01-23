import pandas as pd
from generate_plots.plot_utils_results import read_result, show_result
dataname = 'folktables_2018_mobility_CA'

data_dir = '../results'
##################################################
mix_privga_df = read_result(f'{data_dir}/acs/mix/privga/result_mix_privga.csv')
cat_privga_df = read_result(f'{data_dir}/acs/mix/privga/result_mix_privga.csv')
mix_privga_df['generator'] = 'PrivGA'
cat_privga_df['generator'] = 'PrivGA'
mix_privga_df['queries'] = 'Ranges'
cat_privga_df['queries'] = 'Categorical'
##################################################
gem_df = read_result(f'{data_dir}/acs/mix/gem/gem.csv')
gem_df['generator'] = 'GEM'
##################################################
rap_df = read_result(f'{data_dir}/acs/mix/rap/rap.csv')
rap_df['generator'] = 'RAP'
##################################################

df = pd.concat([privga_df, rap_df, gem_df], ignore_index=True)
df = df.loc[df['data'] == dataname, :]

show_result(df)