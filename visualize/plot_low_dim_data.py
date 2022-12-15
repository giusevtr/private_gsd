import matplotlib.pyplot as plt
import numpy as np

def plot_1d_data(array, title=''):
    plt.title(title)
    plt.hist(array.squeeze(), density=True, alpha=0.5)
    plt.xlim(0, 1)
    plt.show()

def plot_2d_data(data_array, i=0, j=1, alpha=1.0, title='', save_path=None):
    plt.figure(figsize=(5, 5))
    plt.title(title)
    plt.scatter(data_array[:, i], data_array[:, j], alpha=alpha, s=0.7)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()

def plot_2d_data_sync(data_array, sync_data_array,  i=0, j=1, alpha=0.5, title=''):
    plt.figure(figsize=(5, 5))
    plt.title(title)
    plt.scatter(data_array[:, i], data_array[:, j], alpha=alpha, label='real')
    plt.scatter(sync_data_array[:, i], sync_data_array[:, j], alpha=alpha, label='sync')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()
    plt.show()

def plot_digit(digit_arrary: np.array, title=''):
    plt.gray()
    plt.matshow(digit_arrary.reshape(8, 8))
    plt.show()


from sklearn import decomposition
def plot_high_dim_to_2_dim(real, sync, title):
    real = np.array(real)
    sync = np.array(sync)
    n_real, d1 = real.shape
    n_sync, d2 = sync.shape
    assert d1 == d2

    plt.title(f'PCA projection: {title}')
    pca = decomposition.PCA(n_components=2)
    pca.fit(real)
    real_2d = pca.transform(real)
    pca2 = decomposition.PCA(n_components=2)
    pca2.fit(sync)
    sync_2d = pca2.transform(sync)

    # Make them equal size
    rng = np.random.default_rng(seed=0)
    if n_real > n_sync:
        idx = rng.choice(np.arange(n_real), size=n_sync, replace=False)
        real_2d = real_2d[idx]
    elif n_sync > n_real:
        idx = rng.choice(np.arange(n_sync), size=n_real, replace=False)
        sync_2d = sync_2d[idx]

    plt.scatter(real_2d[:, 0], real_2d[:, 1], cmap=plt.cm.nipy_spectral, edgecolor="k",  alpha=0.5, label='real')
    plt.scatter(sync_2d[:, 0], sync_2d[:, 1],  cmap=plt.cm.nipy_spectral, edgecolor="k", alpha=0.5, label='sync')
    plt.legend()
    plt.show()

