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
from experiments.param_search import  param_search


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

    param_search(data, data_name, stats_module)