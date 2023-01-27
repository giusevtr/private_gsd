import pandas as pd
from generate_plots.plot_utils_results import read_result, show_result

error_type = 'l1 error'
res_df = []
for error_type in ['max error', 'l1 error']:
    ##################################################
    privga_df = read_result('../ICML/cat_ranges_results/privga/result_cat_privga.csv', error_type)
    privga_df['generator'] = 'PrivGA'
    gem_df = read_result('../ICML/cat_ranges_results/gem/gem.csv', error_type)
    gem_df['generator'] = 'GEM'
    rap_df = read_result('../ICML/cat_ranges_results/rap/rap.csv', error_type)
    rap_df['generator'] = 'RAP'
    pgm_df = read_result('../ICML/cat_ranges_results/pgm/pgm_em.csv', error_type)
    pgm_df['generator'] = 'PGM'
    df_cat = pd.concat([privga_df, gem_df, rap_df, pgm_df], ignore_index=True)
    df_cat.loc[:, 'Statistics'] = 'Marginals'
    ##################################################



    ##################################################
    privga_df = read_result(f'../ICML/mix_results/privga/result_mix_privga.csv', error_type)
    privga_df['generator'] = 'PrivGA'
    gem_df = read_result(f'../ICML/mix_results/gem/gem.csv', error_type)
    gem_df['generator'] = 'GEM'
    rap_df = read_result(f'../ICML/mix_results/rap/rap.csv', error_type)
    rap_df['generator'] = 'RAP'
    df_mix = pd.concat([privga_df, rap_df, gem_df], ignore_index=True)
    df_mix.loc[:, 'Statistics'] = 'Range'
    ##################################################


    df = pd.concat([df_cat, df_mix], ignore_index=True)
    # df['error type'] = error_type
    res_df.append(df)


df = pd.concat(res_df, ignore_index=True)
dataname = 'folktables_2018_mobility_CA'
df = df.loc[df['data'] == dataname, :]

show_result(df)

