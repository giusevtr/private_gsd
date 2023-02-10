import numpy as np
import pandas as pd
from utils import Dataset, Domain
import matplotlib.pyplot as plt


def get_sparsecat(DATA_SIZE = 100, CAT_SIZE=10):
    size1 = int(DATA_SIZE * 0.5)
    size2 = int(DATA_SIZE * 0.45)
    size3 = DATA_SIZE - size2 - size1

    domain = Domain(['A', 'B', 'C', 'D', 'E'], [CAT_SIZE, CAT_SIZE, CAT_SIZE, CAT_SIZE, CAT_SIZE])

    assert CAT_SIZE>3
    arr1 = np.column_stack([1 * np.ones(size1), 1 * np.ones(size1), np.random.randint(low=0, high=CAT_SIZE, size=size1)])
    arr2 = np.column_stack([2 * np.ones(size2), 2 * np.ones(size2), np.random.randint(low=0, high=CAT_SIZE, size=size2)])
    arr3 = np.column_stack([3 * np.ones(size3), 3 * np.ones(size3), 3 * np.ones(size3)])
    # arr3 = np.random.randint(low=0, high=cat_size, size=size1)
    arr = np.concatenate((arr1, arr2, arr3))


    arr = np.column_stack((arr, np.random.randint(low=0, high=CAT_SIZE, size=DATA_SIZE)))
    arr = np.column_stack((arr, np.random.randint(low=0, high=CAT_SIZE, size=DATA_SIZE)))


    data = Dataset(pd.DataFrame(arr, columns=domain.attrs), domain)
    return data


if __name__ == "__main__":
    data = get_sparsecat(DATA_SIZE=1000)

    plt.hist(data.to_numpy())
    plt.show()


