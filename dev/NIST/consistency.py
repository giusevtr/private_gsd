import jax.random
import pandas as pd
import numpy as np
import jax.numpy as jnp


INDP_CODES = {
    "N": "N",
    "0170":"AGR",
    "0180":"AGR",
    "0190":"AGR",
    "0270":"AGR",
    "0280":"AGR",
    "0290":"AGR" ,
    "0370":"EXT",
    "0380":"EXT",
    "0390":"EXT",
    "0470":"EXT",
    "0490":"EXT",
    "0570" :"UTL",
    "0580" :"UTL",
    "0590" :"UTL",
    "0670" :"UTL",
    "0680" :"UTL",
    "0690" :"UTL",
    "0770" :"CON",
    "1070" :"MFG",
    "1080" :"MFG",
    "1090" :"MFG",
    "1170" :"MFG",
    "1180" :"MFG",
    "1190" :"MFG",
    "1270" :"MFG",
    "1280" :"MFG",
    "1290" :"MFG",
    "1370" :"MFG",
    "1390" :"MFG",
    "1470" :"MFG",
    "1480" :"MFG",
    "1490" :"MFG",
    "1570" :"MFG",
    "1590" :"MFG",
    "1670" :"MFG",
    "1691" :"MFG",
    "1770" :"MFG",
    "1790" :"MFG",
    "1870" :"MFG",
    "1880" :"MFG",
    "1890" :"MFG",
    "1990" :"MFG",
    "2070" :"MFG",
    "2090" :"MFG",
    "2170" :"MFG",
    "2180" :"MFG",
    "2190" :"MFG",
    "2270" :"MFG",
    "2280" :"MFG",
    "2290" :"MFG",
    "2370" :"MFG",
    "2380" :"MFG",
    "2390" :"MFG",
    "2470" :"MFG",
    "2480" :"MFG",
    "2490" :"MFG",
    "2570" :"MFG",
    "2590" :"MFG",
    "2670" :"MFG",
    "2680" :"MFG",
    "2690" :"MFG",
    "2770" :"MFG",
    "2780" :"MFG",
    "2790" :"MFG",
    "2870" :"MFG",
    "2880" :"MFG",
    "2890" :"MFG",
    "2970" :"MFG",
    "2980" :"MFG",
    "2990" :"MFG",
    "3070" :"MFG",
    "3080" :"MFG",
    "3095" :"MFG",
    "3170" :"MFG",
    "3180" :"MFG",
    "3291" :"MFG",
    "3365" :"MFG",
    "3370" :"MFG",
    "3380" :"MFG",
    "3390" :"MFG",
    "3470" :"MFG",
    "3490" :"MFG",
    "3570" :"MFG",
    "3580" :"MFG",
    "3590" :"MFG",
    "3670" :"MFG",
    "3680" :"MFG",
    "3690" :"MFG",
    "3770" :"MFG",
    "3780" :"MFG",
    "3790" :"MFG",
    "3875" :"MFG",
    "3895" :"MFG",
    "3960" :"MFG",
    "3970" :"MFG",
    "3980" :"MFG",
    "3990" :"MFG",
    "4070" :"WHL",
    "4080" :"WHL",
    "4090" :"WHL",
    "4170" :"WHL",
    "4180" :"WHL",
    "4195" :"WHL",
    "4265" :"WHL",
    "4270" :"WHL",
    "4280" :"WHL",
    "4290" :"WHL",
    "4370" :"WHL",
    "4380" :"WHL",
    "4390" :"WHL",
    "4470" :"WHL",
    "4480" :"WHL",
    "4490" :"WHL",
    "4560" :"WHL",
    "4570" :"WHL",
    "4580" :"WHL",
    "4585" :"WHL",
    "4590" :"WHL",
    "4670" :"RET",
    "4680" :"RET",
    "4690" :"RET",
    "4770" :"RET",
    "4780" :"RET",
    "4795" :"RET",
    "4870" :"RET",
    "4880" :"RET",
    "4890" :"RET",
    "4971" :"RET",
    "4972" :"RET",
    "4980" :"RET",
    "4990" :"RET",
    "5070" :"RET",
    "5080" :"RET",
    "5090" :"RET",
    "5170" :"RET",
    "5180" :"RET",
    "5190" :"RET",
    "5275" :"RET",
    "5280" :"RET",
    "5295" :"RET",
    "5370" :"RET",
    "5381" :"RET",
    "5391" :"RET",
    "5470" :"RET",
    "5480" :"RET",
    "5490" :"RET",
    "5570" :"RET",
    "5580" :"RET",
    "5593" :"RET",
    "5670" :"RET",
    "5680" :"RET",
    "5690" :"RET",
    "5790" :"RET",
    "6070" :"TRN",
    "6080" :"TRN",
    "6090" :"TRN",
    "6170" :"TRN",
    "6180" :"TRN",
    "6190" :"TRN",
    "6270" :"TRN",
    "6280" :"TRN",
    "6290" :"TRN",
    "6370" :"TRN",
    "6380" :"TRN",
    "6390" :"TRN",
    "6470" :"INF",
    "6480" :"INF",
    "6490" :"INF",
    "6570" :"INF",
    "6590" :"INF",
    "6670" :"INF",
    "6672" :"INF",
    "6680" :"INF",
    "6690" :"INF",
    "6695" :"INF",
    "6770" :"INF",
    "6780" :"INF",
    "6870" :"FIN",
    "6880" :"FIN",
    "6890" :"FIN",
    "6970" :"FIN",
    "6991" :"FIN",
    "6992" :"FIN",
    "7071" :"FIN",
    "7072" :"FIN",
    "7080" :"FIN",
    "7181" :"FIN",
    "7190" :"FIN",
    "7270" :"PRF",
    "7280" :"PRF",
    "7290" :"PRF",
    "7370" :"PRF",
    "7380" :"PRF",
    "7390" :"PRF",
    "7460" :"PRF",
    "7470" :"PRF",
    "7480" :"PRF",
    "7490" :"PRF",
    "7570" :"PRF",
    "7580" :"PRF",
    "7590" :"PRF",
    "7670" :"PRF",
    "7680" :"PRF",
    "7690" :"PRF",
    "7770" :"PRF",
    "7780" :"PRF",
    "7790" :"PRF",
    "7860" :"EDU",
    "7870" :"EDU",
    "7880" :"EDU",
    "7890" :"EDU",
    "7970" :"MED",
    "7980" :"MED",
    "7990" :"MED",
    "8070" :"MED",
    "8080" :"MED",
    "8090" :"MED",
    "8170" :"MED",
    "8180" :"MED",
    "8191" :"MED",
    "8192" :"MED",
    "8270" :"MED",
    "8290" :"MED",
    "8370" :"SCA",
    "8380" :"SCA",
    "8390" :"SCA",
    "8470" :"SCA",
    "8561" :"ENT",
    "8562" :"ENT",
    "8563" :"ENT",
    "8564" :"ENT",
    "8570" :"ENT",
    "8580" :"ENT",
    "8590" :"ENT",
    "8660" :"ENT",
    "8670" :"ENT",
    "8680" :"ENT",
    "8690" :"ENT",
    "8770" :"SRV",
    "8780" :"SRV",
    "8790" :"SRV",
    "8870" :"SRV",
    "8891" :"SRV",
    "8970" :"SRV",
    "8980" :"SRV",
    "8990" :"SRV",
    "9070" :"SRV",
    "9080" :"SRV",
    "9090" :"SRV",
    "9160" :"SRV",
    "9170" :"SRV",
    "9180" :"SRV",
    "9190" :"SRV",
    "9290" :"SRV",
    "9370" :"ADM",
    "9380" :"ADM",
    "9390" :"ADM",
    "9470" :"ADM",
    "9480" :"ADM",
    "9490" :"ADM",
    "9570" :"ADM",
    "9590" :"ADM",
    "9670" :"MIL",
    "9680" :"MIL",
    "9690" :"MIL",
    "9770" :"MIL",
    "9780" :"MIL",
    "9790" :"MIL",
    "9870" :"MIL",
    "9920" :"UNEMPLOYED"
}
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



def get_consistency_fn(domain, preprocessor, axis=0):

    def get_encoded_value(feature, value):
        # return preprocessor.encoders[feature].transform(np.array(value))
        # df_val = pd.DataFrame([[value]], columns=[feature])
        # return preprocessor.transform_ord(df_val).values[0]
        if feature in preprocessor.attrs_cat:
            enc = preprocessor.encoders[feature]
            value = str(value)
            v = pd.DataFrame(np.array([value]), columns=[feature])
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
    phd_encoded = get_encoded_value('EDU', 12)
    dis_veteran_encoded = get_encoded_value('DVET', 1)
    print(age_15_encoded)
    print(married_status_encoded)
    print()



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
    indp_idx = domain.get_attribute_indices(['INDP']).squeeze().astype(int)
    # indp_cat_idx = domain.get_attribute_indices(['INDP_CAT']).squeeze().astype(int)
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
        has_indp = ~jnp.isnan(x[indp_idx])
        # has_indp_cat = ~jnp.isnan(x[indp_cat_idx])
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
            (is_minor & has_indp),  # Children don't have industry codes
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


def get_nist_all_population_consistency_fn(domain, preprocessor):
    count_inconsistency_fn = get_consistency_fn(domain, preprocessor)
    count_inconsistency_population_fn = jax.jit(jax.vmap(count_inconsistency_fn, in_axes=(0, )))
    return count_inconsistency_population_fn
