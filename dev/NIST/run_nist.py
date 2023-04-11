import os, sys
import jax.random
import pandas as pd
import numpy as np
from models import GeneticSD, GeneticSDV2
from stats import ChainedStatistics,  Marginals, NullCounts, Prefix
import jax.numpy as jnp
from dp_data import load_domain_config, load_df
from dp_data.data_preprocessor import DataPreprocessor
from utils import timer, Dataset, Domain, filter_outliers
import pickle
import matplotlib.pyplot as plt

inc_bins_pre = np.array([-10000,
                         -100,
                         -10,
                         0,
                         # 5,
                         10,
                         50,
                         100,
                         200,
                         500, 700, 1000, 2000, 3000, 4500, 8000, 9000, 10000,
                         10800,
                         12000,
                         14390,
                         15000,
                         18010,
                         20000,
                         23000,
                         25000,
                         27800,
                         30000,
                         33000,
                         35000,
                         37000,
                         40000,
                         45000,
                         47040,
                         50000,
                         55020,
                         60000,
                         65000,
                         67000,
                         70000,
                         75000,
                         80000,
                         85000,
                         90000,
                         95000,
                         100000,
                         101300,
                         140020,
                         200000,
                         300000,
                         400000,
                         500000,
                         1166000])


def get_consistency_fn(domain, preprocessor):

    def get_encoded_value(feature, value):
        # return preprocessor.encoders[feature].transform(np.array(value))
        # df_val = pd.DataFrame([[value]], columns=[feature])
        # return preprocessor.transform_ord(df_val).values[0]
        if feature in preprocessor.attrs_cat:
            enc = preprocessor.encoders[feature]
            value = str(value)
            v = pd.DataFrame(np.array([value]))
            return enc.transform(v)[0]
        if feature in preprocessor.mappings_ord.keys():
            min_val, _ = preprocessor.mappings_ord[feature]
            return value - min_val


    age_15_encoded = get_encoded_value('AGEP', 15)
    married_status_encoded = get_encoded_value('MSP', 4)
    phd_encoded = get_encoded_value('EDU', 12)
    print(age_15_encoded)
    print(married_status_encoded)
    print()



    ## Inconsistensies
    puma_idx = domain.get_attribute_indices(['PUMA']).squeeze().astype(int)
    sex_idx = domain.get_attribute_indices(['SEX']).squeeze().astype(int)
    hisp_idx = domain.get_attribute_indices(['HISP']).squeeze().astype(int)
    rac1p_idx = domain.get_attribute_indices(['RAC1P']).squeeze().astype(int)
    housing_idx = domain.get_attribute_indices(['HOUSING_TYPE']).squeeze().astype(int)
    own_rent_idx = domain.get_attribute_indices(['OWN_RENT']).squeeze().astype(int)
    density_idx = domain.get_attribute_indices(['DENSITY']).squeeze().astype(int)
    deye_idx = domain.get_attribute_indices(['DEYE']).squeeze().astype(int)
    dear_idx = domain.get_attribute_indices(['DEAR']).squeeze().astype(int)
    age_idx = domain.get_attribute_indices(['AGEP']).squeeze().astype(int)
    married_idx = domain.get_attribute_indices(['MSP']).squeeze().astype(int)
    income_idx = domain.get_attribute_indices(['PINCP']).squeeze().astype(int)
    income_decile_idx = domain.get_attribute_indices(['PINCP_DECILE']).squeeze().astype(int)
    indp_idx = domain.get_attribute_indices(['INDP']).squeeze().astype(int)
    indp_cat_idx = domain.get_attribute_indices(['INDP_CAT']).squeeze().astype(int)
    noc_idx = domain.get_attribute_indices(['NOC']).squeeze().astype(int) # Number of children
    npf_idx = domain.get_attribute_indices(['NPF']).squeeze().astype(int) # Family size
    edu_idx = domain.get_attribute_indices(['EDU']).squeeze().astype(int) # Family size
    def row_inconsistency(x: jnp.ndarray):

        is_minor = (x[age_idx] <= age_15_encoded)
        is_married = ~jnp.isnan(x[married_idx])
        has_income = ~jnp.isnan(x[income_idx])
        has_indp = ~jnp.isnan(x[indp_idx])
        has_indp_cat = ~jnp.isnan(x[indp_cat_idx])
        violations = jnp.array([
            jnp.isnan(x[age_idx]), # Age is null
            jnp.isnan(x[puma_idx]),  #
            jnp.isnan(x[sex_idx]),  #
            jnp.isnan(x[hisp_idx]),  #
            jnp.isnan(x[rac1p_idx]),  #
            jnp.isnan(x[housing_idx]),  #
            jnp.isnan(x[own_rent_idx]),  #
            jnp.isnan(x[density_idx]),  #
            jnp.isnan(x[deye_idx]),  #
            jnp.isnan(x[dear_idx]),  #
            (is_minor & is_married),  # Children cannot be married
            (is_minor & has_income),  # Children cannot have income
            (is_minor & (~jnp.isnan(x[income_decile_idx]))),  # Children cannot have income
            jnp.isnan(x[indp_idx]) & (~jnp.isnan(x[indp_cat_idx])),  # Industry codes must match. Either
            (~jnp.isnan(x[indp_idx])) & (jnp.isnan(x[indp_cat_idx])),  # Both are null or non-are null
            (x[noc_idx] >= x[npf_idx]),  # Number of children must be less than family size
            (is_minor & has_indp),  # Children don't have industry codes
            (is_minor & has_indp_cat),  # Children don't have industry codes
            (is_minor & (x[edu_idx] == phd_encoded))  # Children don't have phd
        ])

        return violations
    # Dataset consistency count function
    row_inconsistency_vmap = jax.vmap(row_inconsistency, in_axes=(0, ))
    def count_inconsistency_fn(X):
        inconsistencies = row_inconsistency_vmap(X)
        aggregate = (jnp.sum(inconsistencies, axis=0) / X.shape[0])**2
        return aggregate.sum()
    count_inconsistency_population_fn = jax.jit(jax.vmap(count_inconsistency_fn, in_axes=(0, )))
    return count_inconsistency_population_fn

import numpy as np
if __name__ == "__main__":
    # epsilon_vals = [10]
    epsilon_vals = [10]
    epsilon_vals.reverse()
    dataset_name = 'national2019'
    root_path = '../../dp-data-dev/datasets/preprocessed/sdnist_dce/'
    config = load_domain_config(dataset_name, root_path=root_path)
    df_train = load_df(dataset_name, root_path=root_path)

    domain = Domain(config)
    data = Dataset(df_train, domain)
    preprocessor_path = os.path.join(root_path + dataset_name, 'preprocessor.pkl')

    bins = {}
    with open(preprocessor_path, 'rb') as handle:
        # preprocessor:
        preprocessor = pickle.load(handle)
        temp: pd.DataFrame
        preprocessor: DataPreprocessor
        min_val, max_val = preprocessor.mappings_num['PINCP']
        print(min_val, max_val)
        inc_bins = (inc_bins_pre - min_val) / (max_val - min_val)
        bins['PINCP'] = inc_bins

    # Create statistics and evaluate
    key = jax.random.PRNGKey(0)
    # One-shot queries
    # module0 = Marginals.get_all_kway_combinations(data.domain, k=2, bins=[2, 4, 8, 16, 32, 64])
    module0 = Marginals.get_all_kway_combinations(data.domain, k=2, bins=bins, levels=5)
    stat_module = ChainedStatistics([module0])
    stat_module.fit(data)

    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module._get_workload_fn()

    algo = GeneticSDV2(num_generations=60000,
                       print_progress=True,
                       stop_early=True,
                        domain=data.domain,
                       population_size=100,
                       data_size=2000,
                       inconsistency_fn=get_consistency_fn(domain, preprocessor),
                       mate_perturbation=0
                       )
    # Choose algorithm parameters

    delta = 1.0 / len(data) ** 2
    # Generate differentially private synthetic data with ADAPTIVE mechanism
    for eps in epsilon_vals:
        for seed in [0]:
            key = jax.random.PRNGKey(seed)
            t0 = timer()

            sync_data = algo.fit_dp(key, stat_module=stat_module,
                               epsilon=eps,
                               delta=delta)

            sync_dir = f'sync_data/{dataset_name}/{eps:.2f}/oneshot'
            os.makedirs(sync_dir, exist_ok=True)
            print(f'{sync_dir}/sync_data_{seed}.csv')
            sync_data.df.to_csv(f'{sync_dir}/sync_data_{seed}.csv', index=False)
            errors = jnp.abs(true_stats - stat_fn(sync_data.to_numpy()))
            print(f'GSD(oneshot): eps={eps:.2f}, seed={seed}'
                  f'\t max error = {errors.max():.5f}'
                  f'\t avg error = {errors.mean():.6f}'
                  f'\t time = {timer() - t0:.4f}')

        print()

