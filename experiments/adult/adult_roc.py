import pandas as pd
import os
from models import GSD
from utils import Dataset, Domain
import numpy as np

import jax.random
from models import GeneticSDConsistent as GeneticSD
from models import GSD
from stats import ChainedStatistics,  Marginals, NullCounts
import jax.numpy as jnp
import matplotlib.pyplot as plt

from stats.thresholds import get_thresholds_ordinal

from stats import ChainedStatistics,  Marginals, NullCounts
from dp_data import cleanup, DataPreprocessor, get_config_from_json
QUANTILES = 50

for seed in [0]:

    path = 'dp-data-dev/data2/data/adult/X_num_train.npy'
    X_num_train = np.load(f'../../dp-data-dev/data2/data/adult/X_num_train.npy').astype(int)
    X_num_val = np.load(f'../../dp-data-dev/data2/data/adult/X_num_val.npy').astype(int)
    X_num_test = np.load(f'../../dp-data-dev/data2/data/adult/X_num_test.npy').astype(int)

    X_cat_train = np.load(f'../../dp-data-dev/data2/data/adult/X_cat_train.npy')
    X_cat_val = np.load(f'../../dp-data-dev/data2/data/adult/X_cat_val.npy')
    X_cat_test = np.load(f'../../dp-data-dev/data2/data/adult/X_cat_test.npy')

    y_train = np.load(f'../../dp-data-dev/data2/data/adult/y_train.npy')
    y_val = np.load(f'../../dp-data-dev/data2/data/adult/y_val.npy')
    y_test = np.load(f'../../dp-data-dev/data2/data/adult/y_test.npy')

    cat_cols = [f'cat_{i}' for i in range(X_cat_train.shape[1])]
    num_cols = [f'num_{i}' for i in range(X_num_train.shape[1])]
    all_cols = cat_cols + num_cols + ['Label']

    train_df = pd.DataFrame(np.column_stack((X_cat_train, X_num_train, y_train)), columns=all_cols)
    val_df = pd.DataFrame(np.column_stack((X_cat_val, X_num_val, y_val)), columns=all_cols)
    test_df = pd.DataFrame(np.column_stack((X_cat_test, X_num_test, y_test)), columns=all_cols)
    all_df = pd.concat([train_df, val_df, test_df])

    config = get_config_from_json({'categorical': cat_cols + ['Label'], 'ordinal': num_cols, 'numerical': []})
    preprocessor = DataPreprocessor(config=config)
    preprocessor.fit(all_df)
    pre_train_df = preprocessor.transform(train_df)
    pre_val_df = preprocessor.transform(val_df)
    pre_test_df = preprocessor.transform(test_df)


    N = len(train_df)
    min_bin_size = N // 200 # Number of points on each edge
    bin_edges = {}
    confi_dict = preprocessor.get_domain()
    for col_name in num_cols:
        values = pre_train_df[col_name].values.astype(int)
        sz = confi_dict[col_name]['size']

        bin_edges[col_name] = get_thresholds_ordinal(pre_train_df[col_name], min_bin_size, sz,
                                                     levels=20)
        # plt.title(col_name)
        # plt.hist(values, bins=edges3)
        # plt.show()
        print()


    domain = Domain(confi_dict, bin_edges=bin_edges)
    print()
