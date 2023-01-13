import itertools
import os

import jax.random
import pandas as pd

from models import RelaxedProjection
from stats import Marginals
from utils.utils_data import get_data
from utils.cdp2adp import cdp_rho
from utils import Dataset

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

    # Discretize data using binning strategy
    data_disc = data.discretize(num_bins=32)

    # Get categorical-marginals queries for the discretized data
    train_stats_module = Marginals(data_disc.domain, kway_combinations=kway_combinations)
    train_stats_module.fit(data_disc)

    # This step is for converting the discrete synthetic data back to numerica domain
    numeric_features = data.domain.get_numeric_cols()
    rap_post_processing = lambda data: Dataset.to_numeric(data, numeric_features)


    DATA_SIZE = [100, 1000]
    LR = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02]

    SEEDS = [0, 1, 2]
    EPS = [0.01, 1]

    results = []
    for lr, data_size in itertools.product(LR, DATA_SIZE):

        ########
        # PrivGA
        ########
        # Initialize RAP with discrete domain
        rap = RelaxedProjection(domain=data_disc.domain,
                                iterations=50000,
                                data_size=data_size,
                                learning_rate=lr,
                                print_progress=False)


        for seed, eps in itertools.product(SEEDS, EPS):

            # Generate differentially private synthetic data with ADAPTIVE mechanism
            key = jax.random.PRNGKey(seed)
            N = data.df.shape[0]
            stime = time.time()
            private_stats = train_stats_module.get_private_statistics(key, rho=cdp_rho(eps=eps, delta=1/N**2))
            sync_data_2 = rap.fit(key, stat=private_stats, tolerance=0.0)

            ave_error = private_stats.true_loss_l2(sync_data_2.to_numpy())
            max_error = private_stats.true_loss_inf(sync_data_2.to_numpy())
            ave_error_priv = private_stats.private_loss_l2(sync_data_2.to_numpy())
            max_error_priv = private_stats.private_loss_inf(sync_data_2.to_numpy())
            # print(pop, elite, mut, mate, data_size, seed, ':')
            print(f'lr={lr:.7f}, data_size={data_size:<3}, seed={seed}, eps={eps:.2f}', end='\t\t ')
            print(f'RAP: ave_error = {ave_error:.7f}, max_error={max_error:.5f}, ave_error_priv={ave_error_priv:.7f}, max_error_priv={max_error_priv:.5f} time={time.time() - stime:.5f}')

            results.append([lr, data_size, seed, eps, ave_error, max_error, time.time() - stime])

            res_df = pd.DataFrame(results, columns=['lr', 'data_size', 'seed', 'eps',
                                                    'ave_error', 'max_error', 'time'])
            save_path = f'param_results/{data_name}/RAP'
            os.makedirs(save_path, exist_ok=True)
            save_path = f'{save_path}/rap_params_{seed}.csv'
            res_df.to_csv(save_path, index_label=False)


