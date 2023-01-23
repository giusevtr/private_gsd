import pandas as pd
from generate_plots.plot_utils_results import read_result, show_result


##################################################
privga_df = read_result('../ICML/cat_results/privga/result_cat_privga.csv')
privga_df['generator'] = 'PrivGA'
gem_df = read_result('../ICML/cat_results/gem/gem.csv')
gem_df['generator'] = 'GEM'
rap_df = read_result('../ICML/cat_results/rap/rap.csv')
rap_df['generator'] = 'RAP'
pgm_df = read_result('../ICML/cat_results/pgm/pgm_em.csv')
pgm_df['generator'] = 'PGM'
df_cat = pd.concat([privga_df, gem_df, rap_df, pgm_df], ignore_index=True)
df_cat.loc[:, 'Statistics'] = 'Categorical Marginals'
##################################################



##################################################
privga_df = read_result(f'../ICML/mix_results/privga/result_mix_privga.csv')
privga_df['generator'] = 'PrivGA'
gem_df = read_result(f'../ICML/mix_results/gem/gem.csv')
gem_df['generator'] = 'GEM'
rap_df = read_result(f'../ICML/mix_results/rap/rap.csv')
rap_df['generator'] = 'RAP'
df_mix = pd.concat([privga_df, rap_df, gem_df], ignore_index=True)
df_mix.loc[:, 'Statistics'] = 'Range Marginals'
##################################################


df = pd.concat([df_cat, df_mix], ignore_index=True)
dataname = 'folktables_2018_mobility_CA'
df = df.loc[df['data'] == dataname, :]

show_result(df)

