import jax.random
import pandas as pd
import numpy as np
import jax.numpy as jnp

INDP_CAT = {
        "N": "N",
       "AGR" :"0" ,
       "EXT" :"1" ,
       "UTL" :"2" ,
       "CON" :"3" ,
       "MFG" :"4" ,
       "WHL" :"5" ,
       "RET" :"6" ,
       "TRN" :"7" ,
       "INF" :"8" ,
       "FIN" :"9" ,
       "PRF": "10" ,
       "EDU": "11" ,
       "MED": "12" ,
       "SCA": "13" ,
       "ENT": "14" ,
       "SRV": "15" ,
       "ADM": "16" ,
       "MIL": "17" ,
    "UNEMPLOYED" : "18"
}



def get_nist_simple_consistency_fn(domain, preprocessor, axis=0):

    def get_encoded_value(feature, value):
        if feature in preprocessor.attrs_cat:
            enc = preprocessor.encoders[feature]
            value = str(value)
            v = pd.DataFrame(np.array([value]), columns=[feature]).values.ravel()
            return enc.transform(v)[0]
        if feature in preprocessor.mappings_ord.keys():
            min_val, _ = preprocessor.mappings_ord[feature]
            return value - min_val


    age_15_encoded = get_encoded_value('AGEP', 15)
    age_10_encoded = get_encoded_value('AGEP', 10)
    age_5_encoded = get_encoded_value('AGEP', 5)
    age_3_encoded = get_encoded_value('AGEP', 3)
    dphy_2_encoded = get_encoded_value('DPHY', 2)
    married_status_encoded = get_encoded_value('MSP', 4)
    edu_5_encoded = get_encoded_value('EDU', 5)
    edu_6_encoded = get_encoded_value('EDU', 6)
    edu_7_encoded = get_encoded_value('EDU', 7)
    hs_diploma_encoded = get_encoded_value('EDU', 8)
    edu_9_encoded = get_encoded_value('EDU', 9)
    edu_10_encoded = get_encoded_value('EDU', 10)
    edu_11_encoded = get_encoded_value('EDU', 11)
    phd_encoded = get_encoded_value('EDU', 12)

    ## Inconsistencies
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
    # indp_idx = domain.get_attribute_indices(['INDP']).squeeze().astype(int)
    indp_cat_idx = domain.get_attribute_indices(['INDP_CAT']).squeeze().astype(int)
    noc_idx = domain.get_attribute_indices(['NOC']).squeeze().astype(int) # Number of children
    npf_idx = domain.get_attribute_indices(['NPF']).squeeze().astype(int) # Family size
    edu_idx = domain.get_attribute_indices(['EDU']).squeeze().astype(int) # Education
    dvet_idx = domain.get_attribute_indices(['DVET']).squeeze().astype(int) #
    dphy_idx = domain.get_attribute_indices(['DPHY']).squeeze().astype(int) #   physical disability
    drem_idx = domain.get_attribute_indices(['DREM']).squeeze().astype(int) #   cognitive disability
    msp_idx = domain.get_attribute_indices(['MSP']).squeeze().astype(int) #  Married Status

    housing_type_idx = domain.get_attribute_indices(['HOUSING_TYPE']).squeeze().astype(int) #   cognitive disability
    own_rent_idx = domain.get_attribute_indices(['OWN_RENT']).squeeze().astype(int) #   cognitive disability
    housing_type_1_encoded = get_encoded_value('HOUSING_TYPE', 1)  # House
    housing_type_2_encoded = get_encoded_value('HOUSING_TYPE', 2)  # Group home: Student dorm
    housing_type_3_encoded = get_encoded_value('HOUSING_TYPE', 3)  #
    own_rent_0_encoded = get_encoded_value('OWN_RENT', 0)  #
    own_rent_1_encoded = get_encoded_value('OWN_RENT', 1)  #
    own_rent_2_encoded = get_encoded_value('OWN_RENT', 2)  #

    def row_inconsistency(x: jnp.ndarray):

        is_minor = (x[age_idx] < age_15_encoded)
        is_married = ~jnp.isnan(x[married_idx])
        has_income = ~jnp.isnan(x[income_idx])
        # has_indp = ~jnp.isnan(x[indp_idx])
        has_indp_cat = ~jnp.isnan(x[indp_cat_idx])
        violations = [
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
            (is_minor & has_indp_cat),  # Children don't have industry codes
            (is_minor & (x[edu_idx] == phd_encoded)),  # Children don't have phd
            ((is_minor) & (~jnp.isnan(x[dvet_idx]))),

            # (x[age_idx] < age_10_encoded) & (~jnp.isnan(x[noc_idx])),
            (x[age_idx] < age_5_encoded) & (~jnp.isnan(x[dphy_idx])),
            (x[age_idx] < age_5_encoded) & (~jnp.isnan(x[drem_idx])),
            (x[age_idx] < age_5_encoded) & (x[edu_idx] == edu_5_encoded), # toddler_diploma
            (x[age_idx] < age_5_encoded) & (x[edu_idx] == edu_6_encoded), # toddler_diploma
            (x[age_idx] < age_5_encoded) & (x[edu_idx] == edu_7_encoded), # toddler_diploma
            (x[age_idx] < age_5_encoded) & (x[edu_idx] == hs_diploma_encoded), # toddler_diploma
            (x[age_idx] < age_5_encoded) & (x[edu_idx] == edu_9_encoded), # toddler_diploma
            (x[age_idx] < age_5_encoded) & (x[edu_idx] == edu_10_encoded), # toddler_diploma
            (x[age_idx] < age_5_encoded) & (x[edu_idx] == edu_10_encoded), # toddler_diploma
            (x[age_idx] < age_3_encoded) & (~jnp.isnan(x[edu_idx])), # infant_EDU: Infants (< 3) aren't in school

            (x[age_idx] >= age_15_encoded) & (jnp.isnan(x[dphy_idx]) | jnp.isnan(x[drem_idx]) | jnp.isnan(x[edu_idx]) | jnp.isnan(x[msp_idx]) | jnp.isnan(x[income_idx]) | jnp.isnan(x[income_decile_idx])),

            # Housing-based
            (~jnp.isnan(x[noc_idx])) & (~jnp.isnan(x[npf_idx])) & (x[noc_idx] >= x[npf_idx]) ,  # too_many_children
            (x[housing_type_idx] == housing_type_3_encoded) & (x[own_rent_idx] == own_rent_1_encoded),
            (x[housing_type_idx] == housing_type_2_encoded) & (x[own_rent_idx] == own_rent_2_encoded),
            (x[housing_type_idx] == housing_type_2_encoded) & (x[own_rent_idx] == own_rent_1_encoded),
            (x[housing_type_idx] == housing_type_2_encoded) & (~jnp.isnan(x[npf_idx])), # gq_h_family_NPF: Individuals who live in group quarters aren't considered family households:
            (x[housing_type_idx] == housing_type_2_encoded) & (~jnp.isnan(x[noc_idx])), # gq_h_family_NPF: Individuals who live in group quarters aren't considered family households:
            (x[housing_type_idx] == housing_type_1_encoded) & (jnp.isnan(x[noc_idx])), #  house_NOC: Individuals who live in houses must provide number of children:
            (x[housing_type_idx] == housing_type_1_encoded) & (x[own_rent_idx] == own_rent_0_encoded),
            (x[housing_type_idx] == housing_type_3_encoded) & (x[own_rent_idx] == own_rent_2_encoded),
            (x[housing_type_idx] == housing_type_3_encoded) & (~jnp.isnan(x[npf_idx])),
            (x[housing_type_idx] == housing_type_3_encoded) & (~jnp.isnan(x[noc_idx])),
            # gq_h_family_NPF: Individuals who live in group quarters aren't considered family households:
        ]

        violations = jnp.array(violations)
        return violations
    # Dataset consistency count function
    row_inconsistency_vmap = jax.vmap(row_inconsistency, in_axes=(0, ))
    def count_inconsistency_fn(X):
        inconsistencies = row_inconsistency_vmap(X)
        aggregate = (jnp.sum(inconsistencies, axis=axis) / X.shape[0])
        return aggregate
    # count_inconsistency_population_fn = jax.jit(jax.vmap(count_inconsistency_fn, in_axes=(0, )))
    return count_inconsistency_fn


def get_nist_simple_population_consistency_fn(domain, preprocessor):
    count_inconsistency_fn = get_nist_simple_consistency_fn(domain, preprocessor)
    count_inconsistency_population_fn = jax.jit(jax.vmap(count_inconsistency_fn, in_axes=(0, )))
    return count_inconsistency_population_fn
