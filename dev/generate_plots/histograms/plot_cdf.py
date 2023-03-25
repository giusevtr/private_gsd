import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# sns.set(font_scale=2.0)
sns.set_style("whitegrid")



from utils.utils_data import get_data

data_name = f'folktables_2018_real_CA'
data = get_data(f'{data_name}-mixed-train',
                domain_name=f'domain/{data_name}-mixed', root_path='../../../data_files/folktables_datasets_real')


X_LABEL = 'Threshold'
# cols = ['INTP', 'SEMP']
# cols = ['INTP']
cols = ['WAGP']

feat_label = 'Feature'
gen_label = 'Generator'
type_label = 'Data Type'

real_data_label = 'Real Data'
synthetic_data_label = 'Synthetic Data'

PRIVGA = pd.read_csv('../../sync_data/folktables_2018_acsreal_CA/GSD/Prefix/100/1.00/sync_data_0.csv')
# pd.read_csv('../sync_data/')
RAPpp = pd.read_csv('../../sync_data/folktables_2018_acsreal_CA/RAP++/Prefix/10/1.00/sync_data_0.csv')
RAP = pd.read_csv('../../sync_data/folktables_2018_acsreal_CA/RAP/Ranges/80/1.00/sync_data_0.csv')
df_list = []

BINS = 1000
# brange = 0.20
brange = 1.0
for col in data.domain.get_numeric_cols():
# for col in cols:
    privga_df = PRIVGA[col].copy().to_frame()
    privga_df = privga_df.rename(columns={col: X_LABEL})
    privga_df[feat_label] = col
    privga_df[gen_label] = 'GSD'
    privga_df[type_label] = synthetic_data_label

    rappp_df = RAPpp[col].copy().to_frame()
    rappp_df = rappp_df.rename(columns={col: X_LABEL})
    rappp_df[feat_label] = col
    rappp_df[gen_label] = 'RAP++'
    rappp_df[type_label] = synthetic_data_label

    rap_df = RAP[col].copy().to_frame()
    rap_df = rap_df.rename(columns={col: X_LABEL})
    rap_df[feat_label] = col
    rap_df[gen_label] = 'RAP'
    rap_df[type_label] = synthetic_data_label



    df_real1 = data.df[col].copy().to_frame()
    df_real1 = df_real1.rename(columns={col: X_LABEL})
    df_real1[feat_label] = col
    df_real1[gen_label] = 'GSD'
    df_real1[type_label] = real_data_label

    df_real2 = data.df[col].copy().to_frame()
    df_real2 = df_real2.rename(columns={col: X_LABEL})
    df_real2[feat_label] = col
    df_real2[gen_label] = 'RAP++'
    df_real2[type_label] = real_data_label
    #
    df_real3 = data.df[col].copy().to_frame()
    df_real3 = df_real3.rename(columns={col: X_LABEL})
    df_real3[feat_label] = col
    df_real3[gen_label] = 'RAP'
    df_real3[type_label] = real_data_label


    df_list.append(df_real1)
    df_list.append(df_real2)
    df_list.append(df_real3)
    df_list.append(privga_df)
    df_list.append(rap_df)
    df_list.append(rappp_df)


df = pd.concat(df_list)

def custom_plot(x,  **kwargs):

    # print(kwargs)
    # cumulative = True
    # fill = False
    # if kwargs['label'] == real_data_label:
    #     kwargs['color'] = 'k'
    #     sns.histplot(data=x, bins=BINS, binrange=(0, brange),  stat='density',
    #                  alpha=0.7,
    #                  fill=fill,
    #                  # hatch='/',
    #                  kde=False,
    #                  element="step",
    #                  cumulative=cumulative,
    #                  **kwargs)
    # else:
    #     sns.histplot(data=x, bins=BINS, binrange=(0, brange), stat='density',
    #                  kde=False,
    #                  fill=fill,
    #                  element="step",
    #                  cumulative=cumulative,
    #                  linewidth=3,
    #                  alpha=0.7,
    #                  **kwargs)
    sns.ecdfplot(data=x)

    # plt.plot(x, y, linewidth=3, **kwargs)
    # plt.hist(x, y, s=50, linewidth=4, **kwargs)

df = df[df[feat_label] == 'WKHP']
# sns.ecdfplot(data=df, x='Values',
#              hue=gen_label,
#              # row=feat_label,
#              # style=type_label
#              linewidth=3
#              )
g = sns.FacetGrid(df, col=gen_label,
                  hue=type_label,
                  hue_order=['Real Data', 'Synthetic Data'],
                  # alpha=0.7
                  )
g.map(sns.ecdfplot, X_LABEL, alpha=0.7)
# g = sns.FacetGrid(data=df,
#                   row=feat_label, col=gen_label,
#                   hue=type_label,
#                   sharey='row',
#                   sharex=True,
#                   aspect=2.0,
#                   col_order=['GSD', 'RAP++', 'RAP'],
#                   # col_order=['GSD', 'RAP'],
#                   legend_out=False)
# g.map(custom_plot, 'Values')
# plt.title()
# g.fig.subplots_adjust(top=0.9) # adjust the Figure in rp

plt.subplots_adjust(top=0.75, bottom=0.20)
g.fig.suptitle('CDF plot of WKHP(Proportion of values below a threshold)')

g.add_legend(title='', fontsize=14)
# sns.move_legend(g, "lower center", bbox_to_anchor=(.5, .00),
#                 ncol=2, title=None, frameon=False, fontsize=26)
# sns.move_legend(g, title='Data Type', frameon=False, fontsize=20)
# for ax in g.axes.flat:
#     # ax.set(s=40)
#     title = ax.get_title()
#     algorithm = title.split(' ')[-1]
#     feature = title.split(' ')[2]
#     if algorithm == 'GSD':
#         ax.set_title(f'{algorithm}', fontsize=28, weight='bold')
#     else:
#         ax.set_title(f'{algorithm}', fontsize=28)
#
#     print(ax)
#     # ax.set_xlabel(rf'Feature Values', fontsize=26)
#     ax.set_ylabel(f'{feature}', fontsize=26)
#     ax.set_xlabel(f'', fontsize=26)
#     ax.set_yticklabels([])
#     ax.tick_params(axis='y', which='major', labelsize=16, rotation=0)
#     ax.tick_params(axis='x', which='major', labelsize=20, rotation=10)


plt.show()