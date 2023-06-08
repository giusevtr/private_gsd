import pandas as pd
import numpy as np

df_real = pd.read_csv('real_train.csv')
df_sync = pd.read_csv('sync_data_50_1000.csv')



n = len(df_real)
m = len(df_sync)

dist = np.zeros((n, m))
for i in range(n):
    for j in range(m):
        x_real = df_real.values[i, :]
        x_sync = df_sync.values[j, :]
        # print(x_real)
        l2_dist = np.linalg.norm(x_real - x_sync)
        dist[i, j] = l2_dist

        # print(f'dist[{i}, {j}] = {dist[i, j]}')


dist = dist.min(axis=1)

print(dist)

