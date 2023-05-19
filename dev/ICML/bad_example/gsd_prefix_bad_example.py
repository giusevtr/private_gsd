import itertools
import jax.random
import pandas as pd
import os
from models import PrivGA, SimpleGAforSyncData
from stats import ChainedStatistics
import jax.numpy as jnp
from utils import timer, Dataset, Domain
import matplotlib.pyplot as plt
from stats.prefix_fixed import Prefix
import numpy as np
if __name__ == "__main__":
    module_name = 'Prefix'

    domain = Domain(['A'], [1])
    data_np = np.ones(30000) * 0.5


    data = Dataset(pd.DataFrame(data_np, columns=['A']), domain)


    module = Prefix(domain,
                        k_cat=0,
                        cat_kway_combinations=[],
                        k_prefix=1,
                        pre_columns=np.array([0, 0, 0]),
                    thresholds=np.array([0.49, 0.5, 0.51])
                    )

    stat_module = ChainedStatistics([module])
    stat_module.fit(data)
    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module.get_dataset_statistics_fn()

    stat_test = stat_fn(data)
    print(stat_test)

    algo = PrivGA(num_generations=100000, domain=domain, data_size=1000, population_size=100)

    sync_data = algo.fit_dp(key=jax.random.PRNGKey(0), stat_module=stat_module, epsilon=1, delta=1e-6)

    sync_stats = stat_fn(sync_data)

    errors = jnp.abs(stat_test - sync_stats)
    print(f'max error = {errors.max()}')
    sync_data.df['A'].hist()
    plt.title('GSD')
    plt.xlim([0, 0.6])
    plt.show()


    sync_data.df.to_csv('gsd_badexample.csv', index=False)

    data.df['A'].hist()
    plt.title('Original')
    plt.show()






