import numpy as np
import pandas as pd
from utils import Dataset, Domain
import matplotlib.pyplot as plt


def get_sparsecat(DATA_SIZE = 100):
    size1 = int(DATA_SIZE * 0.5)
    size2 = int(DATA_SIZE * 0.45)
    size3 = DATA_SIZE - size2 - size1
    cat_size = 100

    domain = Domain(['A', 'B', 'C'], [cat_size, cat_size, cat_size])

    arr1 = np.column_stack([10 * np.ones(size1), 10 * np.ones(size1), np.random.randint(low=0, high=cat_size, size=size1)])
    arr2 = np.column_stack([20 * np.ones(size2), 20 * np.ones(size2), np.random.randint(low=0, high=cat_size, size=size2)])
    arr3 = np.column_stack([30 * np.ones(size3), 30 * np.ones(size3), 3 * np.ones(size3)])
    arr = np.concatenate((arr1, arr2, arr3))

    data = Dataset(pd.DataFrame(arr, columns=domain.attrs), domain)
    return data


if __name__ == "__main__":
    data = get_sparsecat(DATA_SIZE=1000)

    plt.hist(data.to_numpy())
    plt.show()


