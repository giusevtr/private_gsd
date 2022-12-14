import folktables
import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSEmployment
from utils import Dataset, Domain, DataTransformer
import sys, os

CONT_FEATURES = [
               'AGEP',  # cont
               'WKHP',  # Numeric:Usual hours worked per week past 12 months
               'PWGTP',  # Numeric: Housing Unit Weight
               'PWGTP1',  # Numeric:
               'INTP',  # Numeric: Interest, dividends
               'JWMNP',  # Numeric:Travel time to work
               'JWRIP',  # Numeric: Vehicle occupancy
               'OIP',  # Numeric: All other income past 12 months
               'PAP',  # Numeric: Public assistance income past 12 months
               'RETP',  # Numeric: Retirement income past 12 months
               'SEMP',  # Numeric: Self-employment income past 12 months
               'SSIP',  # Numeric: Supplementary Security Income past 12 months
               'WAGP',  # Numeric: Wages or salary income past 12 months
               'PERNP',  # Numeric: PERNP
               'PINCP',  # Numeric: Total person's income
               'POVPIP',  # Numeric: Income-to-poverty ratio
           ]

def get_dataset_name(task, state):

    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"], download=True)
    features, label, group = ACSEmployment.df_to_numpy(acs_data)
    print(ACSEmployment.features)

    cat_features = []
    con_features = []


    for col in ACSEmployment.features:
        if col in CONT_FEATURES:
            con_features.append(col)
        else:
            cat_features.append(col)

    transformer = DataTransformer()
    domain = transformer.fit(acs_data, discrete_columns=cat_features, real_columns=con_features)

    df = pd.DataFrame(features, ACSEmployment.features)

    data_np = transformer.transform(df)

    data_df = pd.DataFrame(data_np, ACSEmployment.features)
    data = Dataset(data_df, domain=domain)

    return f'{task}_{state}', data

# def run_experiments(data:Dataset)
def run_experiments(data_name, algo_name, epsilon):



    save_path = 'results'
    os.makedirs(save_path, exist_ok=True)
    save_path =os.path.join(save_path, data_name)
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, algo_name)
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f'{epsilon:.2f}')
    os.makedirs(save_path, exist_ok=True)

    print(save_path)


if __name__ == "__main__":
    # df = folktables.

    get_dataset_name('income', 'CA')
