import numpy as np
import jax.numpy as jnp
from utils import Dataset, Domain


def train_ML(X, label_col):
    pass


def train_DS(X, domain, label_col, epsilon, iterations: int = 10, sync_data_size: int =100, seed:int = 0):

    rng = np.random.default_rng(seed)

    sync_data = Dataset.synthetic_rng(domain, sync_data_size, rng)
    X_sync = sync_data.to_numpy()

    ml_stats = []
    for it in range(iterations):

        model = train_ML(X_sync, label_col)


        # Save answers
        ml_stats.append(model.get_error(X, label_col))



