from sklearn.datasets import make_circles, load_digits
from utils import Dataset, Domain
import numpy as np
import pandas as pd

def get_digits_dataset():
    X, y = load_digits(return_X_y=True)
    D = np.column_stack((X/16.0, y))
    columns = [f'f{i}' for i in range(64)] + ['label']
    shape = [1 for _ in range(64)] + [10]
    df = pd.DataFrame(D, columns=columns)
    data = Dataset(df=df, domain=Domain(columns, shape))
    return data
