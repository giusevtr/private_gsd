import os
import jax.random
import pandas as pd

from models import GeneticSD, GeneticStrategy
from stats import ChainedStatistics,  Marginals, NullCounts, Prefix
import jax.numpy as jnp
from dp_data import load_domain_config, load_df
from utils import timer, Dataset, Domain, filter_outliers
import pickle


if __name__ == "__main__":
    # epsilon_vals = [10]
    epsilon_vals = [10]

    dataset_name = 'national2019'
    root_path = '../../../dp-data-dev/datasets/preprocessed/sdnist_dce/'
    config = load_domain_config(dataset_name, root_path=root_path)
    df_train = load_df(dataset_name, root_path=root_path)

    domain = Domain(config)
    data = Dataset(df_train, domain)

    # Construct inconsistency function
    age_idx = domain.get_attribute_indices(['AGEP']).squeeze()
    married_idx = domain.get_attribute_indices(['MSP']).squeeze()
    income_idx = domain.get_attribute_indices(['PINCP']).squeeze()
    def row_inconsistency(x: jnp.ndarray):
        num_violations = 0
        is_minor = x[age_idx] < 15
        is_married = x[married_idx] == 4
        has_income = ~(x[income_idx] == jnp.nan)
        num_violations += (is_minor & is_married)  # Children cannot be married
        num_violations += (is_minor & has_income)  # Children cannot have income
        return num_violations

    # Dataset consistency count function
    row_inconsistency_vmap = jax.vmap(row_inconsistency, in_axes=(0, ))
    def count_inconsistency_fn(X):
        inconsistencies = row_inconsistency_vmap(X)
        return jnp.sum(inconsistencies)

    # Population(of synthetic data sets) consistency count function
    count_inconsistency_population_fn = jax.vmap(count_inconsistency_fn, in_axes=(0, ))

    ############
    # Test consistency function
    ############

    # Get a single dataset and count inconsistencies
    data_size = 1000
    sync = Dataset.synthetic(domain, N=data_size, seed=0, null_values=0.1)
    print(f'Sync Data inconsistencies count:')
    print(count_inconsistency_fn(sync.to_numpy()))

    # Get a bunch of synthetic datasets and count the number of inconsistencies of each
    d = len(domain.attrs)
    population_size = 10
    rng = jax.random.PRNGKey(0)
    pop = Dataset.synthetic_jax_rng(domain, population_size * data_size, rng, null_values=0.1)
    sd_population = pop.reshape((population_size, data_size, d))

    population_inconsistencies = count_inconsistency_population_fn(sd_population)
    print(f'Population inconsistencies:')
    print(population_inconsistencies)