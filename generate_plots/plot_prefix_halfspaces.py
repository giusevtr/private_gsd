import pandas as pd
from generate_plots.plot_utils_results import read_result, show_result

error_type = 'max error'
##################################################
privga_hs_df = read_result('../ICML/prefix_halfspaces_results/result_mix_privga_halfspaces.csv', error_type)
privga_hs_df['generator'] = 'PrivGA'
privga_hs_df['Statistics'] = 'Halfspaces'
privga_pr_df = read_result('../ICML/prefix_halfspaces_results/result_mix_privga_prefix.csv', error_type)
privga_pr_df['Statistics'] = 'Prefixes'
privga_pr_df['generator'] = 'PrivGA'


rap_prefix_df = read_result('../ICML/prefix_halfspaces_results/result_mix_RAP_Prefix.csv', error_type)
rap_prefix_df['Statistics'] = 'Prefixes'
rap_prefix_df['generator'] = 'RAP'


gem_prefix_df = read_result('../ICML/prefix_halfspaces_results/result_mix_GEM_Prefix.csv', error_type)
gem_prefix_df['Statistics'] = 'Prefixes'
gem_prefix_df['generator'] = 'GEM'


rap_hs_df = read_result('../ICML/prefix_halfspaces_results/result_mix_RAP_Halfspaces.csv', error_type)
rap_hs_df['Statistics'] = 'Halfspaces'
rap_hs_df['generator'] = 'RAP'

gem_hs_df = read_result('../ICML/prefix_halfspaces_results/result_mix_GEM_Halfspaces.csv', error_type)
gem_hs_df['Statistics'] = 'Halfspaces'
gem_hs_df['generator'] = 'GEM'

rappp_hs_df = read_result('../ICML/prefix_halfspaces_results/results_mix_RAP++_Halfspaces.csv', error_type)
rappp_hs_df['Statistics'] = 'Halfspaces'
rappp_hs_df['generator'] = 'RAP++'
#####

df = pd.concat([
   privga_hs_df,
   privga_pr_df,
   rap_hs_df, rap_prefix_df,
   gem_hs_df, gem_prefix_df,
   rappp_hs_df
], ignore_index=True)
dataname = 'folktables_2018_real_CA'
df = df.loc[df['data'] == dataname, :]

show_result(df, error_type)

