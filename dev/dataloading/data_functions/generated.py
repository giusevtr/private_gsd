import numpy as np
import pandas as pd
# from dev.dataloading.dataset import Dataset
# from dev.dataloading.domain import Domain
from utils import Dataset, Domain
from dev.dataloading.transformer import Transformer
import sklearn
from sklearn.datasets import make_classification, make_moons
from dev.dataloading.data_functions.data_container import DatasetContainer


def get_moons(n=20000):
    def moons_fn(seed):
        min_max_scaler = sklearn.preprocessing.MinMaxScaler()
        X, y = make_moons(n_samples=n, random_state=seed, noise=0.05)
        y = y.reshape(-1, 1).astype(int)
        attrs = ["f0", "f1"] + ["label"]
        shape = [1, 1, 2]

        X_train_minmax = min_max_scaler.fit_transform(X)
        # y_train_minmax = min_max_scaler.fit_transform(y)

        domain = Domain(attrs, shape, ["label"])
        df = pd.DataFrame(np.hstack((X_train_minmax, y)), columns=attrs)

        dataset = Dataset(df, domain)
        train, test = dataset.split(0.8)
        from_df_to_dataset = lambda df: Dataset(df, domain)

        from_dataset_to_df = lambda dataset: dataset.df
        return DatasetContainer(
            "moons",
            train,
            test,
            from_df_to_dataset,
            from_dataset_to_df,
            cat_columns=["label"],
            num_columns=["f0", "f1"],
            label=["label"],
        )

    return moons_fn


def get_classification(n=20000, d=2, num_bins=None):
    def classification_fn(seed):
        min_max_scaler = sklearn.preprocessing.MinMaxScaler()
        X, y = make_classification(
            n_samples=n,
            n_features=d,
            n_informative=d,
            n_redundant=0,
            class_sep=0.9,
            random_state=seed,
        )
        y = y.reshape(-1, 1)
        attrs = [f"f{i}" for i in range(d)] + ["label"]
        shape = [1 for _ in range(d)] + [2]
        targets = ["label"]
        X_train_minmax = min_max_scaler.fit_transform(X)
        df = pd.DataFrame(np.hstack((X_train_minmax, y)), columns=attrs)

        cat_cols = ["label"]
        num_cols = [f"f{i}" for i in range(d)]

        for cat in cat_cols:
            df[cat] = df[cat].astype(int)

        transformer = Transformer(
            cat_cols, num_cols, bin_size=num_bins, normalize=False
        )
        transformer.fit(df)
        dataset_all_rows = transformer.transform(df, target=["label"])

        train, test = dataset_all_rows.split(0.8, seed=seed)
        from_df_to_dataset = lambda df: transformer.transform(df, targets)
        from_dataset_to_df = lambda dataset: transformer.inverse_transform(dataset)
        return DatasetContainer(
            f"classification_{d}d",
            train,
            test,
            from_df_to_dataset,
            from_dataset_to_df,
            cat_columns=cat_cols,
            num_columns=num_cols,
            label=targets,
        )

        # domain = Domain(attrs, shape, ['label'])
        # from_df_to_dataset = lambda df: Dataset(df, domain)
        # from_dataset_to_df = lambda dataset: dataset.df
        #
        # dataset = Dataset(df, domain)
        # train, test = dataset.split(0.8)
        # return DatasetContainer(f'classification_{d}d', train, test, from_df_to_dataset, from_dataset_to_df,
        #                         cat_columns=['label'],
        #                         num_columns=[f'f{i}' for i in range(d)],
        #                         label=['label']
        #                         )

    return classification_fn
