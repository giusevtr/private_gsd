import sys, os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dp_data import load_domain_config, load_df
from utils import timer, Dataset, Domain, filter_outliers

def plot_density(df_orig, sync, feature_name, range):
    ## PLOT PINCP
    cutoff = range
    # sync = df.sample(n=len(df_orig), replace=True)
    real = df_orig[df_orig[feature_name] < cutoff][feature_name].to_frame()
    real['Type'] = 'Real'
    temp = sync[sync[feature_name] < cutoff][feature_name].to_frame()
    temp['Type'] = 'Sync'
    df_income = pd.concat([real, temp])
    sns.histplot(data=df_income, x=feature_name, hue='Type', bins=100)
    plt.show()


dataset_name = sys.argv[1]  # options: national2019, tx2019, ma2019
sync_path = sys.argv[2]
save_post_pat = sys.argv[3]

print(sync_path)
df = pd.read_csv(sync_path)

root_path = '../../dp-data-dev/datasets/preprocessed/sdnist_dce/'
config = load_domain_config(dataset_name, root_path=root_path)
df_orig = load_df(dataset_name, root_path=root_path)
preprocessor_path = os.path.join(root_path + dataset_name, 'preprocessor.pkl')
with open(preprocessor_path, 'rb') as handle:
    # preprocessor:
    preprocessor = pickle.load(handle)
    temp: pd.DataFrame
    temp = preprocessor.inverse_transform(df_orig)
    print(temp)


plot_density(df_orig.sample(n=2000), df, 'DENSITY', range=1)
plot_density(df_orig.sample(n=2000), df, 'WGTP', range=0.05)
