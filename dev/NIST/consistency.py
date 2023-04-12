import jax.random
import pandas as pd
import numpy as np
import jax.numpy as jnp


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
    age_10_encoded = get_encoded_value('AGEP', 10)
    age_5_encoded = get_encoded_value('AGEP', 5)
    dphy_2_encoded = get_encoded_value('DPHY', 2)
    married_status_encoded = get_encoded_value('MSP', 4)
    phd_encoded = get_encoded_value('EDU', 12)
    dis_veteran_encoded = get_encoded_value('DVET', 1)
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
    edu_idx = domain.get_attribute_indices(['EDU']).squeeze().astype(int) # Education
    dvet_idx = domain.get_attribute_indices(['DVET']).squeeze().astype(int) #
    dphy_idx = domain.get_attribute_indices(['DPHY']).squeeze().astype(int) #   physical disability
    drem_idx = domain.get_attribute_indices(['DREAM']).squeeze().astype(int) #   cognitive disability
    def row_inconsistency(x: jnp.ndarray):

        is_minor = (x[age_idx] < age_15_encoded)
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
            (is_minor & has_indp),  # Children don't have industry codes
            (is_minor & has_indp_cat),  # Children don't have industry codes
            (is_minor & (x[edu_idx] == phd_encoded)),  # Children don't have phd
            (is_minor) & (x[dvet_idx] == dis_veteran_encoded),
            # (x[age_idx] < age_10_encoded) & (~jnp.isnan(x[noc_idx])),
            # (x[age_idx] < age_5_encoded) & (~jnp.isnan(x[dphy_idx])),
            # (x[age_idx] < age_5_encoded) & (~jnp.isnan(x[drem_idx])),
        ])

        return violations
    # Dataset consistency count function
    row_inconsistency_vmap = jax.vmap(row_inconsistency, in_axes=(0, ))
    def count_inconsistency_fn(X):
        inconsistencies = row_inconsistency_vmap(X)
        aggregate = (jnp.sum(inconsistencies, axis=0) / X.shape[0])
        return aggregate
    # count_inconsistency_population_fn = jax.jit(jax.vmap(count_inconsistency_fn, in_axes=(0, )))
    return count_inconsistency_fn


def get_population_consistency_fn(domain, preprocessor):
    count_inconsistency_fn = get_consistency_fn(domain, preprocessor)
    count_inconsistency_population_fn = jax.jit(jax.vmap(count_inconsistency_fn, in_axes=(0, )))
    return count_inconsistency_population_fn
