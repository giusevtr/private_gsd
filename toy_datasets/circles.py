from sklearn.datasets import make_circles
from utils import Dataset, Domain

def get_circles_dataset(DATA_SIZE = 100, noise=0.03, seed=0):
    X, y = make_circles(n_samples=DATA_SIZE, noise=noise, random_state=seed)
    x_max = X.max(axis=0)
    x_min = X.min(axis=0)
    X = (X - x_min) / (x_max - x_min)
    return Dataset.from_onehot_to_dataset(Domain(['A', 'B'], [1, 1]), X)
