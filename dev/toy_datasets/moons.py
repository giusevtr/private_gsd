from sklearn.datasets import make_moons
from utils import Dataset, Domain
import numpy as np
import pandas as pd

def get_moons_dataset(DATA_SIZE = 100, noise=0.03, seed=0):
    X, y = make_moons(n_samples=DATA_SIZE, noise=noise, random_state=seed)
    x_max = X.max(axis=0)
    x_min = X.min(axis=0)
    X = (X - x_min) / (x_max - x_min)
    X = np.column_stack((X, y))

    df = pd.DataFrame(X, columns=['A', 'B', 'C'])
    data = Dataset(df=df, domain=Domain(['A', 'B', 'C'], [1, 1, 2]))
    return data
