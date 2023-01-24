import pandas as pd
from generate_plots.plot_utils_results import read_result, show_result

error_type = 'l1 error'
##################################################
privga_hs_df = read_result('../ICML/figure_2/result_mix_privga_halfspaces.csv', error_type)
privga_hs_df['generator'] = 'PrivGA'
privga_hs_df['Statistics'] = 'Halfspaces'
privga_pr_df = read_result('../ICML/figure_2/result_mix_privga_prefix.csv', error_type)
privga_pr_df['Statistics'] = 'Prefixes'
privga_pr_df['generator'] = 'PrivGA'


#
# gem_df = read_result('../ICML/cat_results/gem/gem.csv')
# gem_df['generator'] = 'GEM'
# rap_df = read_result('../ICML/cat_results/rap/rap.csv')
# rap_df['generator'] = 'RAP'
# pgm_df = read_result('../ICML/cat_results/pgm/pgm_em.csv')
# pgm_df['generator'] = 'PGM'
# df_cat = pd.concat([privga_df, gem_df, rap_df, pgm_df], ignore_index=True)
# df_cat.loc[:, 'Statistics'] = 'Categorical Marginals'
##################################################



##################################################
# privga_df = read_result(f'../ICML/mix_results/privga/result_mix_privga.csv')
# privga_df['generator'] = 'PrivGA'
# gem_df = read_result(f'../ICML/mix_results/gem/gem.csv')
# gem_df['generator'] = 'GEM'
# rap_df = read_result(f'../ICML/mix_results/rap/rap.csv')
# rap_df['generator'] = 'RAP'
# df_mix = pd.concat([privga_df, rap_df, gem_df], ignore_index=True)
# df_mix.loc[:, 'Statistics'] = 'Range Marginals'
##################################################


df = pd.concat([
   privga_hs_df,
   privga_pr_df
], ignore_index=True)
dataname = 'folktables_2018_real_CA'
df = df.loc[df['data'] == dataname, :]

show_result(df, error_type)

