import itertools
import os
import jax.random
import pandas as pd
from models import PrivGA, SimpleGAforSyncData
from stats import Marginals
from utils.utils_data import get_data
from utils.cdp2adp import cdp_rho
import time


if __name__ == "__main__":

    # Get Data
    task = 'income'
    state = 'CA'
    data_name = f'folktables_2018_{task}_{state}'
    data = get_data(f'folktables_datasets/{data_name}-mixed-train',
                    domain_name=f'folktables_datasets/domain/{data_name}-num', root_path='../../data_files')
    data, col_range = data.normalize_real_values()
    stats_module, kway_combinations = Marginals.get_all_kway_mixed_combinations(data.domain, k_disc=0, k_real=2,
                                                                                bins=[2, 4, 8, 16, 32])

    stats_module.fit(data)

    POP = [200, 2000]  # Population size
    ELITE = [2, 10]  # Elite set size
    MUT = [1, 5]  # Number of mutations per generation
    MATE_RATE = [0, 0.05]  # Cross mating rate (0 or 5%)
    SIZE = [100, 1000] # Synthetic Data Size
    SEEDS = [0, 1, 2]
    EPS = [0.01, 1]

    results = []
    for pop, elite, mut, mate_rate, data_size in itertools.product(POP, ELITE, MUT, MATE_RATE, SIZE):
        mate = int(mate_rate * data_size)

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

        for seed, eps in itertools.product(SEEDS, EPS):

            # Generate differentially private synthetic data with ADAPTIVE mechanism
            key = jax.random.PRNGKey(seed)
            N = data.df.shape[0]
            stime = time.time()
            private_stats = stats_module.get_private_statistics(key, rho=cdp_rho(eps=eps, delta=1/N**2))
            sync_data_2 = priv_ga.fit(key, stat=private_stats, tolerance=0.0)

            ave_error = private_stats.true_loss_l2(sync_data_2.to_numpy())
            max_error = private_stats.true_loss_inf(sync_data_2.to_numpy())
            ave_error_priv = private_stats.priv_loss_l2(sync_data_2.to_numpy())
            max_error_priv = private_stats.priv_loss_inf(sync_data_2.to_numpy())
            # print(pop, elite, mut, mate, data_size, seed, ':')
            print(f'pop={pop:<3}, elite={elite:<3}, mut={mut:<3}, mate={mate:<3}, data size={data_size:<5}, seed={seed}, eps={eps:.2f}', end='\t\t ')
            print(f'PrivGA: ave_error = {ave_error:.7f}, max_error={max_error:.5f}, ave_error_priv={ave_error_priv:.7f}, max_error_priv={max_error_priv:.5f} time={time.time() - stime:.5f}')

            results.append([pop, elite, mut, mate, data_size, seed, eps, ave_error, max_error, time.time() - stime])

            res_df = pd.DataFrame(results, columns=['pop', 'elite', 'mut', 'mate', 'data_size', 'seed', 'eps',
                                                    'ave_error', 'max_error', 'time'])

            save_path = f'param_results/{data_name}/PrivGA'
            os.makedirs(save_path)
            save_path = f'{save_path}/privga_params_{seed}.csv'
            res_df.to_csv(save_path, index_label=False)


