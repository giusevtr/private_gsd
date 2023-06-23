import os
import pandas as pd
from dev.dataloading.transformer import Transformer
from dev.dataloading.data_functions.data_container import DatasetContainer


def get_bank(full=True, bin_size=None):
    name = "bank"
    path = os.path.dirname(__file__)
    if full:
        df = pd.read_csv(f"{path}/../../data_raw/bank-full.csv", delimiter=",")
        name = "bank_full"
    else:
        df = pd.read_csv(f"{path}/../../data_raw/bank.csv", delimiter=",")

    cat_cols = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "poutcome",
        "y",
    ]  # 11
    con_cols = [
        "age",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    ]  # 10

    transformer = Transformer(cat_cols, con_cols, bin_size=bin_size, normalize=True)
    # raw_data_combined_df = raw_data_df.append(raw_data_test_df, ignore_index=True)
    transformer.fit(df)

    def bank_fn(seed):
        train_df = df.sample(frac=0.8, random_state=seed)

        # Creating dataframe with
        # rest of the 50% values
        test_df = df.drop(train_df.index)

        train_dataset = transformer.transform(train_df, ["label"])
        test_dataset = transformer.transform(test_df, ["label"])
        # post_df = transformer.inverse_transform(dataset)

        post_fn = lambda dataset: transformer.inverse_transform(dataset)
        return DatasetContainer(name, train_dataset, test_dataset, post_fn, label=["y"])

    return bank_fn
