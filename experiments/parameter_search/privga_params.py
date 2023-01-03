import itertools

import jax.random
import pandas as pd

from models import PrivGA, SimpleGAforSyncData
from stats import Marginals
from utils.utils_data import get_data
from utils.cdp2adp import cdp_rho

import time
if __name__ == "__main__":

    # Get Data
    ROUNDS = 1
    BINS=32

    # task = 'mobility'
    # state = 'CA'
    # data_name = f'folktables_{task}_2018_{state}'
    # data = get_data(f'folktables_datasets/{data_name}-mixed',
    #                 domain_name=f'folktables_datasets/{data_name}-cat',  root_path='../../data_files/')

    # data = get_data(f'folktables_datasets/{data_name}-mixed',
    #                 domain_name=f'folktables_datasets/{data_name}-num',  root_path='../../data_files/')

    # data = get_data(f'folktables_datasets/{data_name}-mixed',
    #                 domain_name=f'folktables_datasets/{data_name}-mixed',  root_path='../../data_files/')
    data = get_data('adult', 'adult-small', root_path='../../data_files/')

    # Create statistics and evaluate
    marginal_module = Marginals.get_all_kway_mixed_combinations(data.domain, k=2, bins=[2, 4, 8, 16, 32])
    marginal_module.fit(data)


    POP = [20, 50, 100, 150, 200]
    ELITE = [2, 3, 4, 5]
    MUT = [1, 2, 3, 4]
    MATE = [0, 1, 5, 10, 15, 20]
    SIZE = [200]
    SEEDS = [0, 1, 2, 3, 4]

    results = []
    for pop, elite, mut, mate, data_size, seed in itertools.product(POP, ELITE, MUT, MATE, SIZE, SEEDS):

        ########
        # PrivGA
        ########
        priv_ga = PrivGA(
            num_generations=50000,
            stop_loss_time_window=50000,
            print_progress=False,
            strategy=SimpleGAforSyncData(
                domain=data.domain,
                data_size=data_size,
                population_size=pop,
                elite_size=elite,
                muta_rate=mut,
                mate_rate=mate
            )
        )

        # Generate differentially private synthetic data with ADAPTIVE mechanism
        key = jax.random.PRNGKey(seed)
        N = data.df.shape[0]
        stime = time.time()
        private_stats = marginal_module.get_private_statistics(key, rho=cdp_rho(eps=10, delta=1/N**2))
        sync_data_2 = priv_ga.fit(key, stat=private_stats, tolerance=0.030)

        ave_error = private_stats.true_loss_l2(sync_data_2.to_numpy())
        max_error = private_stats.true_loss_inf(sync_data_2.to_numpy())
        ave_error_priv = private_stats.priv_loss_l2(sync_data_2.to_numpy())
        max_error_priv = private_stats.priv_loss_inf(sync_data_2.to_numpy())
        # print(pop, elite, mut, mate, data_size, seed, ':')
        print(f'pop={pop:<3}, elite={elite:<3}, mut={mut:<3}, mate={mate:<3}, data size={data_size:<3}, seed={seed}', end='\t\t ')
        print(f'PrivGA: ave_error = {ave_error:.7f}, max_error={max_error:.5f}, ave_error_priv={ave_error_priv:.7f}, max_error_priv={max_error_priv:.5f} time={time.time() - stime:.5f}')

        results.append([pop, elite, mut, mate, data_size, seed, ave_error, max_error, time.time() - stime])

        res_df = pd.DataFrame(results, columns=['pop', 'elite', 'mut', 'mate', 'data_size', 'seed', 'ave_error', 'max_error', 'time'])
        res_df.to_csv(f'results_{seed}.csv', index_label=False)


