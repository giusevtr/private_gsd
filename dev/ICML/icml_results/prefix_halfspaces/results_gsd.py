import pandas as pd


# df = pd.read_csv('gsd_prefix_2.csv')
df = pd.read_csv('gsd_halfspace.csv')

df = df[df['error type'] == 'Average']
# df = df[df['error type'] == 'Max']
df2 = df.groupby(['Data', 'epsilon', 'error type']).mean()[['error']]
print(df2)


# print('RAP++')
# error_lbl = 'max'
# # error_lbl = 'ave'
# df_rp = pd.read_csv('rap++_results/mix_results/acs_CA_coverage/RAP(Marginal&Halfspace)/prefix/result.csv')
# # df_rp = df_rp.groupby(['dataset', 'epsilon', 'params', 'max']).mean()
# df_rp = df_rp.groupby(['dataset_name', 'param', 'epsilon', ], as_index=False)[error_lbl].mean()
# df_rp = df_rp.groupby(['dataset_name', 'epsilon'], as_index=False)[error_lbl].min()
# print(df_rp)
