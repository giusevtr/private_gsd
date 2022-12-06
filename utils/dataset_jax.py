import json
import jax.nn
import numpy as np
import jax.numpy as jnp
from jax import random
import pandas as pd
from utils import Domain
from sklearn.preprocessing import OneHotEncoder


def get_data_onehot(data):
    df_data = data.df
    dim = sum(data.domain.shape)
    i = 0
    oh_encoded = []
    for attr, num_classes in zip(data.domain.attrs, data.domain.shape):
        col = df_data[attr].values
        if num_classes > 1:
            # Categorical features
            col_oh = jax.nn.one_hot(col.astype(int), num_classes)
            oh_encoded.append(col_oh)
        elif num_classes == 1:
            # Numerical features
            oh_encoded.append(col.astype(float).reshape((-1, 1)))
    data_onehot = jnp.concatenate(oh_encoded, axis=1)

    return data_onehot.astype(float)


class Dataset:
    def __init__(self, df, domain):
        """ create a Dataset object

        :param df: a pandas dataframe
        :param domain: a domain object
        """
        assert set(domain.attrs) <= set(df.columns), 'data must contain domain attributes'
        self.domain = domain
        self.df = df.loc[:, domain.attrs]

    def __len__(self):
        return len(self.df)


    @staticmethod
    def synthetic_jax_rng(domain, N, rng):
        """
        Generate synthetic data conforming to the given domain
        :param domain: The domain object
        :param N: the number of individuals
        """
        d = len(domain.shape)
        rng_split = jax.random.split(rng, d)
        arr = [jax.random.randint(rng_temp, shape=(N, ), minval=0, maxval=n) if n > 1 else
               jax.random.uniform(rng_temp, shape=(N, )) for (rng_temp, n) in zip(rng_split, domain.shape)]
        values = jnp.array(arr).T
        return values

    @staticmethod
    def synthetic_rng(domain, N, rng):
        """ Generate synthetic data conforming to the given domain

        :param domain: The domain object
        :param N: the number of individuals
        """
        arr = [rng.integers(low=0, high=n, size=N) if n>1 else rng.uniform(size=N) for n in domain.shape]
        values = np.array(arr).T
        df = pd.DataFrame(values, columns=domain.attrs)
        return Dataset(df, domain)

    @staticmethod
    def synthetic(domain, N, seed: int):
        rng = np.random.default_rng(seed)
        return Dataset.synthetic_rng(domain, N, rng)

    @staticmethod
    def load(path, domain):
        """ Load data into a dataset object

        :param path: path to csv file
        :param domain: path to json file encoding the domain information
        """
        df = pd.read_csv(path)
        config = json.load(open(domain))
        domain = Domain(config.keys(), config.values())
        return Dataset(df, domain)
    
    def project(self, cols):
        """ project dataset onto a subset of columns """
        if type(cols) in [str, int]:
            cols = [cols]
        data = self.df.loc[:,cols]
        domain = self.domain.project(cols)
        return Dataset(data, domain)

    def drop(self, cols):
        proj = [c for c in self.domain if c not in cols]
        return self.project(proj)

    def datavector(self, flatten=True, weights=None, density=False):
        """ return the database in vector-of-counts form """
        bins = [range(n+1) for n in self.domain.shape]
        ans = jnp.histogramdd(self.df.values, bins, weights=weights, density=density)[0]
        return ans.flatten() if flatten else ans

    def sample(self, p=None, n=None, replace=False, seed=None):
        subsample = None
        if p is not None:
            subsample = self.df.sample(frac=p, replace=replace, random_state=seed)
        elif n is not None:
            subsample = self.df.sample(n=n, replace=replace, random_state=seed)
        return Dataset(subsample, domain=self.domain)

    def even_split(self, k=5, seed=0):
        key = random.PRNGKey(seed)
        datasets = []
        index = jnp.array(list(self.df.index))
        random.shuffle(key, index)
        df_split = jnp.array_split(index, k)
        for kth in df_split:
            df = self.df.loc[jnp.array(kth), :].copy()
            datasets.append(Dataset(df, domain=self.domain))
        return datasets


    def get_row(self, row_index):
        row_df = self.df.iloc[[row_index]]
        return Dataset(row_df, domain=self.domain)

    def get_row_dataset_list(self):
        N = len(self.df)
        res = []
        for k in range(N):
            res.append(self.get_row(k))
        return res


    @staticmethod
    def from_numpy_to_dataset(domain, X):
        df = pd.DataFrame(X, columns=domain.attrs)
        return Dataset(df, domain=domain)

    def to_numpy(self):
        cols = [self.df.values[:, i].astype(int) if n > 1 else self.df.values[:, i].astype(float) for i, n in enumerate(self.domain.shape)]
        df_numpy = np.vstack(cols).T
        return df_numpy

    def to_onehot(self) -> jnp.ndarray:
        df_data = self.df
        oh_encoded = []
        for attr, num_classes in zip(self.domain.attrs, self.domain.shape):
            col = df_data[attr].values
            if num_classes > 1:
                # Categorical features
                col_oh = jax.nn.one_hot(col.astype(int), num_classes)
                oh_encoded.append(col_oh)
            elif num_classes == 1:
                # Numerical features
                oh_encoded.append(col.astype(float).reshape((-1, 1)))
        data_onehot = jnp.concatenate(oh_encoded, axis=1)

        return data_onehot.astype(float)

    @staticmethod
    def from_onehot_to_dataset(domain: Domain, X_oh):

        dtypes = []
        start = 0
        output = []
        column_names = domain.attrs
        cat_cols = domain.get_categorical_cols()
        for col in domain.attrs:
            dimensions = domain.size(col)
            column_onehot_encoding = X_oh[:, start:start + dimensions]


            if col in cat_cols:
                encoder = OneHotEncoder()
                # encoder.categories_ = [list(range(domain.size(col)))]
                dummy = [[v] for v in range(domain.size(col))]
                encoder.fit(dummy)
                col_values = encoder.inverse_transform(column_onehot_encoding)
                dtypes.append('int64')
            else:
                col_values = column_onehot_encoding.squeeze()
                dtypes.append('float')

            output.append(col_values)
            start += dimensions

        data_np = np.column_stack(output)

        data_df = pd.DataFrame(data_np, columns=column_names)

        dtypes = pd.Series(dtypes, data_df.columns)

        data_df = data_df.astype(dtypes)

        data = Dataset(data_df, domain=domain)

        return data


    @staticmethod
    def apply_softmax(domain: Domain, X_relaxed: jnp.ndarray) -> jnp.ndarray:
        # Takes as input relaxed dataset
        # Then outputs a dataset consistent with data schema self.domain
        X_softmax = []
        i = 0
        for attr, num_classes in zip(domain.attrs, domain.shape):
            logits = X_relaxed[:, i:i+num_classes]
            i += num_classes

            if num_classes > 1:
                X_col = jax.nn.softmax(x=logits, axis=1)
                X_softmax.append(X_col)
            else:
                X_softmax.append(logits)


        X_softmax = jnp.concatenate(X_softmax, axis=1)
        return X_softmax

    @staticmethod
    def normalize_categorical(domain: Domain, X_relaxed: jnp.ndarray) -> jnp.ndarray:
        # Takes as input relaxed dataset
        # Then outputs a dataset consistent with data schema self.domain
        X_softmax = []
        i = 0
        for attr, num_classes in zip(domain.attrs, domain.shape):
            logits = X_relaxed[:, i:i+num_classes]
            i += num_classes

            if num_classes > 1:
                # logits = logits < 0.0001
                # min_r = jnp.min(jnp.array([0, jnp.min(logits)]))
                # min_r = jnp.min(logits)
                # logits = logits - min_r

                sum_r = jnp.sum(logits, axis=1)
                temp = jnp.array([sum_r, 0.00001 * jnp.ones_like(sum_r)])
                sum_r = jnp.max(temp, axis=0)
                X_col = logits / sum_r.reshape(-1, 1)
                X_softmax.append(X_col)
            else:
                # maxval = logits.max()
                # minval = logits.min()
                # range_vals = maxval - minval
                # logits_norm = (logits + minval) / range_vals

                X_softmax.append(logits)


        X_softmax = jnp.concatenate(X_softmax, axis=1)
        return X_softmax

    @staticmethod
    def get_sample_onehot(key, domain, X_relaxed: jnp.ndarray, num_samples=1) -> jnp.ndarray:

        keys = jax.random.split(key, len(domain.attrs))
        X_onehot = []
        i = 0
        for attr, num_classes, subkey in zip(domain.attrs, domain.shape, keys):
            logits = X_relaxed[:, i:i+num_classes]
            i += num_classes
            if num_classes > 1:
                row_one_hot = []
                for _ in range(num_samples):

                    sum_r = jnp.sum(logits, axis=1)
                    temp = jnp.array([sum_r, 0.00001 * jnp.ones_like(sum_r)])
                    sum_r = jnp.max(temp, axis=0)
                    logits = logits / sum_r.reshape(-1, 1)

                    subkey, subsubkey = jax.random.split(subkey, 2)
                    categories = jax.random.categorical(subsubkey, jnp.log(logits), axis=1)
                    onehot_col = jax.nn.one_hot(categories.astype(int), num_classes)
                    row_one_hot.append(onehot_col)
                X_onehot.append(jnp.concatenate(row_one_hot, axis=0))
            else:
                row_one_hot = []
                # Add numerical column
                for _ in range(num_samples):
                    row_one_hot.append(logits)

                X_onehot.append(jnp.concatenate(row_one_hot, axis=0))


        X_onehot = jnp.concatenate(X_onehot, axis=1)
        return X_onehot



##################################################
# Test
##################################################

def test_onehot_encoding():
    # data_size = 10
    # import numpy as np
    # rng = np.random.default_rng(9)
    cols = ['A', 'B', 'C']
    dom = Domain(cols, [3, 1, 4])

    raw_data_array = pd.DataFrame([
                        [0, 0.0, 1],
                        [0, 0.1, 3],
                        [1, 0.2, 1]], columns=cols)
    # data = Dataset.synthetic_rng(dom, data_size, rng)
    data = Dataset(raw_data_array, dom)
    print(f'Initial data in original format')
    print(data.df)

    data_oh = data.to_onehot()
    assert data_oh.shape == (3, 8)
    print(f'Onehot dataset')
    print(data_oh)

    data_2 = Dataset.from_onehot_to_dataset(dom, data_oh)

    print(f'Final data in original format')
    print(data_2.df)

    # assert raw_data_array

if __name__ == "__main__":
    test_onehot_encoding()