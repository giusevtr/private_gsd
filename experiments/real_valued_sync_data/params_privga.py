import itertools
import os
import jax.random
import pandas as pd
from models import PrivGAfast,SimpleGAforSyncDataFast
from stats import Marginals
from utils.utils_data import get_data
from utils.cdp2adp import cdp_rho
import time
import jax.numpy as jnp


if __name__ == "__main__":

    # Get Data
    task = 'mobility'
    state = 'CA'
    data_name = f'folktables_2018_{task}_{state}'
    data = get_data(f'folktables_datasets/{data_name}-mixed-train',
                    domain_name=f'folktables_datasets/domain/{data_name}-num', root_path='../../data_files')
    stats_module, kway_combinations = Marginals.get_all_kway_mixed_combinations(data.domain, k_disc=0, k_real=2,
                                                                                bins=[2, 4, 8, 16, 32])

    stats_module.fit(data)
    stats_size = stats_module.get_true_stats().shape[0]
    print(f'stats_size={stats_size}')

    POP = [200, 2000]  # Population size
    ELITE = [2, 10]  # Elite set size
    MUT = [1, 5]  # Number of mutations per generation
    MATE_RATE = [0.01]  # Cross mating rate (0 or 1%)
    SIZE = [200, 2000]  # Synthetic Data Size
    SEEDS = [0, 1, 2]
    EPS = [0.01, 1]
    # EPS = [0.01]

    results = []
    for pop, elite, mut, mate_rate, data_size in itertools.product(POP, ELITE, MUT, MATE_RATE, SIZE):
        mate = int(mate_rate * data_size)

        ########
        # PrivGA
        ########
        priv_ga = PrivGAfast(
            num_generations=500000,
            print_progress=False,
            strategy=SimpleGAforSyncDataFast(
                domain=data.domain,
                data_size=data_size,
                population_size=pop,
                elite_size=elite,
                muta_rate=mut,
                mate_rate=mate
            )
        )

        for eps, seed in itertools.product(EPS, SEEDS):

            # Generate differentially private synthetic data with ADAPTIVE mechanism
            key = jax.random.PRNGKey(seed)
            N = data.df.shape[0]
            stime = time.time()
            # private_stats = stats_module.get_private_statistics(key, rho=cdp_rho(eps=eps, delta=1/N**2))
            sync_data_2 = priv_ga.fit_dp(key, stat_module=stats_module, epsilon=eps, delta=1/N**2)

            errors = jnp.abs(stats_module.get_true_stats() - stats_module.get_stats_jit(sync_data_2))
            max_error = float(errors.max())
            ave_error = float(jnp.linalg.norm(errors, ord=1) / errors.shape[0])

            # print(pop, elite, mut, mate, data_size, seed, ':')
            print(f'pop={pop:<3}, elite={elite:<3}, mut={mut:<3}, mate={mate:<3}, data size={data_size:<5}, eps={eps:.2f}, seed={seed}', end='\t\t ')
            print(f'PrivGA: ave_error = {ave_error:.7f}, max_error={max_error:.5f} time={time.time() - stime:.5f}')

            results.append([pop, elite, mut, mate, data_size, seed, eps, ave_error, max_error, time.time() - stime])

            res_df = pd.DataFrame(results, columns=['pop', 'elite', 'mut', 'mate', 'data_size', 'seed', 'eps',
                                                    'ave_error', 'max_error', 'time'])

            save_path = f'param_results/{data_name}/PrivGA'
            os.makedirs(save_path, exist_ok=True)
            save_path = f'{save_path}/privga_params_{seed}.csv'
            res_df.to_csv(save_path, index_label=False)


