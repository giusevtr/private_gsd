import pandas as pd
from generate_plots.plot_utils_results import read_result, show_result


##################################################
privga_df = read_result('acs/cat/privga/result_cat_privga.csv')
privga_df['generator'] = 'PrivGA'
##################################################

gem_df = read_result('acs/cat/gem/gem.csv')
gem_df['generator'] = 'GEM'
##################################################

rap_df = read_result('acs/cat/rap/rap.csv')
rap_df['generator'] = 'RAP'
##################################################

pgm_df = read_result('acs/cat/pgm/pgm_em.csv')
pgm_df['generator'] = 'PGM'
##################################################


df = pd.concat([privga_df, gem_df, rap_df, pgm_df], ignore_index=True)

dataname = 'folktables_2018_mobility_CA'
df = df.loc[df['data'] == dataname, :]


show_result(df)

