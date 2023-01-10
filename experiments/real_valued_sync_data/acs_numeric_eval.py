import itertools
import folktables
import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSEmployment
from utils import Dataset, Domain, DataTransformer
from models import Generator, PrivGA, SimpleGAforSyncData, RelaxedProjection
from stats import Marginals
from utils.utils_data import get_data
from experiments.experiment import run_experiments
import jax.numpy as jnp
import os
import jax
EPSILON = (0.07, 0.23, 0.52, 0.74, 1.0)
adaptive_rounds = (1, 2, 3, 4, 5)

if __name__ == "__main__":
    # df = folktables.

    tasks = ['employment', 'coverage', 'income', 'mobility', 'travel']
    # tasks = ['employment']
    states = ['CA']

    columns = ['data', 'generator','T', 'epsilon', 'seed', 'max error', 'l1 error']


    RES = []
    for task, state in itertools.product(tasks, states):
        data_name = f'folktables_2018_{task}_{state}'

        orig_data = get_data(f'folktables_datasets/{data_name}-mixed-train',
                             domain_name=f'folktables_datasets/domain/{data_name}-num', root_path='../../data_files')
        orig_data_normed, _ = orig_data.normalize_real_values()
        stats_module, kway_combinations = Marginals.get_all_kway_mixed_combinations(orig_data.domain,
                                                                                    k_disc=0, k_real=2,
                                                                                                    bins=[
                                                                                                        2, 4, 8, 16,
                                                                                                          32])
        stats_module.fit(orig_data_normed)
        orig_stats = stats_module.get_true_stats()

        print()
        print(data_name)
        loc = f'results/{data_name}'
        for root, dirs, files in os.walk(loc):
            for name in files:
                if name.endswith((".csv")):
                    path_list = root.split('/')
                    # print(path_list)

                    algo = path_list[2]
                    T = int(path_list[3])
                    eps = float(path_list[4])

                    seed = int(name[-5])

                    sync_data_path = os.path.join(root, name)
                    # print(f'loading {sync_data_path}')
                    df_sync = pd.read_csv(sync_data_path)
                    sync_data = Dataset(df_sync, domain=orig_data.domain)
                    # sync_data = get_data(sync_data_path,
                    #         domain_name=f'../../data_files/folktables_datasets/domain/{data_name}-num', root_path='.')

                    sync_stats = stats_module.get_stats(sync_data)

                    max_error = float(jnp.abs(orig_stats - sync_stats).max())
                    l1_error = float(jnp.linalg.norm(orig_stats-sync_stats, ord=1)/orig_stats.shape[0])

                    res = [data_name, algo, T, eps, seed, max_error, l1_error]
                    print(res)
                    RES.append(res)



    res_df = pd.DataFrame(RES, columns=columns)
    res_df.to_csv('acs_numeric_results.csv', index=False)

