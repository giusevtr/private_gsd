import pandas as pd
from sklearn.datasets import make_classification
from utils import Dataset, Domain
import numpy as np


def get_classification(DATA_SIZE=100, d=2, seed=0):
    X, y = make_classification(n_samples=DATA_SIZE, n_features=d, n_informative=d, n_redundant=0,
                               n_repeated=0, random_state=seed)
    x_max = X.max(axis=0)
    x_min = X.min(axis=0)
    X = (X - x_min) / (x_max - x_min)
    # arr = np.concatenate([X, y.reshape(-1, 1)], axis=1)
    arr = np.column_stack((X, y))
    cols = [f'f{i}' for i in range(d)] + ['label']
    domain = Domain(cols, [1 for _ in range(d)] + [2])
    df = pd.DataFrame(arr, columns=cols)
    data = Dataset(df, domain)
    return data
# def get_classification(DATA_SIZE=100, d=2, seed=0):
#     X, y = make_classification(n_samples=DATA_SIZE, n_features=d, n_informative=d, n_redundant=0,
#                                n_repeated=0, random_state=seed)
#     x_max = X.max(axis=0)
#     x_min = X.min(axis=0)
#     X = (X - x_min) / (x_max - x_min)
#     arr = np.concatenate([X, y.reshape(-1, 1)], axis=1)
#     cols = [f'f{i}' for i in range(d)] + ['label']
#     domain = Domain(cols, [1 for _ in range(d)] + [2])
#     df = pd.DataFrame(arr, columns=cols)
#     data = Dataset(df, domain)
#     return data