import os
import pandas as pd
from dev.dataloading.transformer import Transformer
from dev.dataloading.data_functions.data_container import DatasetContainer

# from  benchmark.visualize import visualize_dataset

import matplotlib.pyplot as plt
import seaborn as sns


def visualize_dataset(sync_df, label, title="", downsample=False):
    # print(f'data_df.shape = {data_df.shape}')
    sync_df = (
        sync_df.drop("Unnamed: 0", axis=1)
        if "Unnamed: 0" in sync_df.columns
        else sync_df
    )
    # data_df = data_df.drop('index')
    g = sns.PairGrid(sync_df, hue=label, diag_sharey=False, corner=True)
    g.map_diag(sns.histplot)
    g.map_lower(sns.scatterplot, alpha=0.3)
    g.add_legend()
    g.fig.subplots_adjust(top=0.95)
    g.fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def get_cardiovascular(bin_size=None):
    name = "cardio"
    path = os.path.dirname(__file__)
    df = pd.read_csv(f"{path}/../../../data_raw/cardio.csv", delimiter=";")
    print("ap_hi: ", df["ap_hi"].min(), df["ap_hi"].max())
    print("ap_lo: ", df["ap_lo"].min(), df["ap_lo"].max())
    print()
    label = "cardio"
    cat_cols = ["gender", "cholesterol", "gluc", "smoke", "alco", "active", label]
    num_cols = ["age", "height", "weight", "ap_hi", "ap_lo"]

    df = df[df["ap_hi"] <= 300]
    df = df[df["ap_hi"] >= 0]
    df = df[df["ap_lo"] <= 300]
    df = df[df["ap_lo"] >= 0]

    # visualize_dataset(df[con_cols + [label]].sample(500), label=label, title='Original')
    # print(f'dataset size = {len(df)}')

    transformer = Transformer(cat_cols, num_cols, bin_size=bin_size, normalize=True)
    # raw_data_combined_df = raw_data_df.append(raw_data_test_df, ignore_index=True)
    transformer.fit(df)

    def cardio_fn(seed):
        train_df = df.sample(frac=0.8, random_state=seed)

        # Creating dataframe with
        # rest of the 50% values
        test_df = df.drop(train_df.index)

        train_dataset = transformer.transform(train_df, [label])
        test_dataset = transformer.transform(test_df, [label])
        # post_df = transformer.inverse_transform(dataset)
        from_df_to_dataset = lambda df: transformer.transform(df, [label])
        post_fn = lambda dataset: transformer.inverse_transform(dataset)
        return DatasetContainer(
            name,
            train_dataset,
            test_dataset,
            from_df_to_dataset,
            post_fn,
            cat_columns=cat_cols,
            num_columns=num_cols,
            label=[label],
        )

    return cardio_fn


if __name__ == "__main__":
    fn = get_cardiovascular()

    data = fn(0)

    true_dataset_post_df = data.from_dataset_to_df_fn(data.train)
    true_test_dataset_post_df = data.from_dataset_to_df_fn(data.test)

    df = true_dataset_post_df.append(true_test_dataset_post_df, ignore_index=True)

    print("ap_hi", df["ap_hi"].min(), df["ap_hi"].max())
    print("ap_lo", df["ap_lo"].min(), df["ap_lo"].max())
    print()

    # post = data.from_dataset_to_df_fn(data.train)
    #
    # print(post['ap_hi'].min())
    # print(post['ap_hi'].max())
