import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_real = pd.read_csv('real_train.csv')
# df_sync = pd.read_csv('sync_data_5.csv')

n = len(df_real)
m = len(df_sync)

real_target = df_real.pop('label').values
real_feats = df_real.values
for i in range(n):
    img = real_feats[i, :].reshape(28, 28)
    label = real_target[i]
    plt.imshow(img)

    save_path = f'images/{label}/real/img_{i}.png'
    plt.savefig(save_path)

