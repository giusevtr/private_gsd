import numpy as np
import pandas as pd
from sklearn.datasets import  make_blobs
from utils import Dataset, Domain
# from utils.plot_low_dim_data import plot_2d_data


def get_sparse_1d_dataset(DATA_SIZE = 100,  seed=0):
    center1 = int(DATA_SIZE * 0.95)
    center2 = int(DATA_SIZE * 0.04)
    center3 = int(DATA_SIZE * 0.01)
    X, y = make_blobs(n_samples=[center1, center2, center3],
                      cluster_std=[0.01, 0.2, 0.1],
                      random_state=seed)
    x_max = X.max(axis=0) + 0.2
    x_min = X.min(axis=0) - 0.2
    X = (X - x_min) / (x_max - x_min)
    return Dataset.from_onehot_to_dataset(Domain(['A'], [1]), X[:,0])

def get_sparse_dataset(DATA_SIZE = 100,  seed=0):
    center1 = int(DATA_SIZE * 0.60)
    center2 = int(DATA_SIZE * 0.30)
    center3 = int(DATA_SIZE * 0.10)
    X, y = make_blobs(n_samples=[center1, center2, center3],
                      cluster_std=[0.01, 0.2, 0.1],
                      random_state=seed)
    x_max = X.max(axis=0) + 0.2
    x_min = X.min(axis=0) - 0.2
    X = (X - x_min) / (x_max - x_min)

    D = np.column_stack((X, y))

    df = pd.DataFrame(D, columns=['A', 'B', 'C'])
    data = Dataset(df=df, domain=Domain(['A', 'B', 'C'], [1, 1, 3]))
    return data
    # return X

if __name__ == "__main__":
    data =get_sparse_dataset(DATA_SIZE=1000)

