import numpy as np
import pandas as pd
from dev.dataloading.transformer import Transformer
import folktables
from folktables import (
    ACSDataSource,
    ACSEmployment,
    ACSTravelTime,
    ACSIncome,
    ACSMobility,
    ACSPublicCoverage,
)
from dev.dataloading.data_functions.data_type_dict import (
    NUMERIC,
    CATEGORICAL,
    FEATURE_TYPES,
)
from dev.dataloading.data_functions.data_container import DatasetContainer


def add_continuous(ACS):
    feats = ACS.features
    continuous_cols = [
        "WKHP",
        "PWGTP",
        "INTP",
        "JWMNP",
        "JWRIP",
        "PAP",
        "SEMP",
        "WAGP",
        "WAOB",  # World area of birth
        "FOCCP",  # Occupation allocation flag
        "POVPIP",
    ]
    for con in continuous_cols:
        if con not in feats and con != ACS.target:
            feats.append(con)
    # print('debug')
    # print(feats, ACS.target)
    return feats


def remove(arr, rm_list):
    for rm in rm_list:
        if rm in arr:
            arr.remove(rm)
    return arr


def create_new_target(ACSTarget: folktables.BasicProblem, remove_columns: list):
    """Create a new target by removing high cardinality columns and adding continuous columns."""
    ACSTargetV2 = folktables.BasicProblem(
        features=remove(add_continuous(ACSTarget), remove_columns),
        target=ACSTarget.target,
        target_transform=ACSTarget.target_transform,
        group=ACSTarget.group,
        preprocess=ACSTarget._preprocess,
        postprocess=ACSTarget._postprocess,
    )
    return ACSTargetV2


ACSIncomeV2 = create_new_target(ACSIncome, remove_columns=["POBP", "OCCP", "ST"])
ACSEmploymentV2 = create_new_target(ACSEmployment, remove_columns=["ST"])
ACSPublicCoverageV2 = create_new_target(ACSPublicCoverage, remove_columns=["ST"])
ACSTravelTimeV2 = create_new_target(
    ACSTravelTime, remove_columns=["PUMA", "POWPUMA", "OCCP", "ST"]
)
ACSMobilityV2 = create_new_target(ACSMobility, remove_columns=["ST"])


def get_acs_all(state="NY", num_bins=None):
    postprocess = lambda x: np.nan_to_num(x, -1)  # Replace nan with 0
    data_source = ACSDataSource(survey_year=2014, horizon="1-Year", survey="person")
    acs_data = data_source.get_data(states=[state], download=True)

    TASKS = [
        ACSIncomeV2,
        ACSEmploymentV2,
        ACSPublicCoverageV2,
        ACSTravelTimeV2,
        ACSMobilityV2,
    ]
    features = (
        ACSIncomeV2.features
        + ACSEmploymentV2.features
        + ACSPublicCoverageV2.features
        + ACSTravelTimeV2.features
        + ACSMobilityV2.features
    )
    features = list(set(features))  # remove duplicates

    targets = [
        ACSIncomeV2.target,
        ACSEmploymentV2.target,
        ACSPublicCoverageV2.target,
        ACSTravelTimeV2.target,
        ACSMobilityV2.target,
    ]

    features = remove(features, targets)  # remove targets from features columns

    res = []
    for feature in features:
        res.append(acs_data[feature].to_numpy())
    res_array = postprocess(np.column_stack(res))

    target_res = []
    for task, target in zip(TASKS, targets):
        # res.append(acs_data[feature].to_numpy())
        if task.target_transform is None:
            target_arr = acs_data[target].to_numpy()
        else:
            target_arr = task.target_transform(acs_data[target]).to_numpy()

        target_res.append(target_arr)
    target_res_array = np.column_stack(target_res)

    # X, y, group = ACSTarget.df_to_numpy(acs_data)

    data_np = np.column_stack((res_array, target_res_array))
    df = pd.DataFrame(data_np, columns=features + targets)

    # print(ACSTarget.features)
    cat_cols = [col for col in features if FEATURE_TYPES[col] == CATEGORICAL] + targets
    num_cols = [col for col in features if FEATURE_TYPES[col] == NUMERIC]

    for cat in cat_cols:
        df[cat] = df[cat].astype(int)

    transformer = Transformer(cat_cols, num_cols, bin_size=num_bins, normalize=True)
    # data_np = np.hstack((X, y.reshape(-1, 1)))
    transformer.fit(df)
    dataset_all_rows = transformer.transform(df)
    domain = dataset_all_rows.domain
    print(state, end=": ")
    print('features: ', features)
    print('cat features: ', cat_cols)
    print('num features: ', num_cols)
    print("cat size=", len(cat_cols), end=". ")
    print("num size=", len(num_cols), end=". ")
    print(f"dataset size={len(acs_data)}, oh dim={sum(dataset_all_rows.domain.shape)}")

    def acs_fn(seed):
        train, test = dataset_all_rows.split(0.8, seed=seed)
        from_df_to_dataset = lambda df: transformer.transform(df)
        from_dataset_to_df = lambda dataset: transformer.inverse_transform(dataset)
        return DatasetContainer(
            f"acs_{state}",
            train,
            test,
            from_df_to_dataset,
            from_dataset_to_df,
            cat_columns=cat_cols,
            num_columns=num_cols,
            label=targets,
        )

    return dataset_all_rows, acs_fn


def get_acs(state="NY", target="income", num_bins=None):
    data_source = ACSDataSource(survey_year=2014, horizon="1-Year", survey="person")
    acs_data = data_source.get_data(states=[state], download=True)

    ACSTarget = None
    if target == "income":
        ACSTarget = ACSIncomeV2
    if target == "travel":
        ACSTarget = ACSTravelTimeV2
    if target == "employment":
        ACSTarget = ACSEmploymentV2
    if target == "mobility":
        ACSTarget = ACSMobilityV2
    if target == "coverage":
        ACSTarget = ACSPublicCoverageV2
    print("target = ", target)
    X, y, group = ACSTarget.df_to_numpy(acs_data)

    # print(ACSTarget.features)
    cat_cols = [
        col for col in ACSTarget.features if FEATURE_TYPES[col] == CATEGORICAL
    ] + [ACSTarget.target]
    num_cols = [col for col in ACSTarget.features if FEATURE_TYPES[col] == NUMERIC]

    transformer = Transformer(cat_cols, num_cols, bin_size=num_bins, normalize=True)

    data_np = np.hstack((X, y.reshape(-1, 1)))
    df = pd.DataFrame(data_np, columns=ACSTarget.features + [ACSTarget.target])

    transformer.fit(df)
    dataset_all_rows = transformer.transform(df, [ACSTarget.target])
    print(f"state={state}, target={target}: ", end=" ")
    print("cat size=", len(cat_cols), end=". ")
    print("num size=", len(num_cols), end=". ")
    print(f"dataset size = {len(df)}, oh size = {sum(dataset_all_rows.domain.shape)}")

    # print(dataset_all_rows.df['OCCP'].unique())
    def acs_fn(seed):
        train, test = dataset_all_rows.split(0.8, seed=seed)
        from_df_to_dataset = lambda df: transformer.transform(df, [ACSTarget.target])
        from_dataset_to_df = lambda dataset: transformer.inverse_transform(dataset)
        return DatasetContainer(
            f"acs_{state}_{target}",
            train,
            test,
            from_df_to_dataset,
            from_dataset_to_df,
            cat_columns=cat_cols,
            num_columns=num_cols,
            label=[ACSTarget.target],
        )

    return acs_fn


if __name__ == "__main__":
    # acs_fn = get_acs(target='income', state='CA')
    # acs_fn = get_acs(target='travel', state='CA')
    # acs_fn = get_acs(target='employment', state='CA')
    # acs_fn = get_acs(target='mobility', state='CA')

    for st in ["NY", "CA", "TX", "FL", "PA"]:
        for task in ["income", "travel", "employment", "mobility", "coverage"]:
            print(st, task)
            acs_fn = get_acs(state=st, target=task)
            data = acs_fn(0)

    # true_dataset_post_df = data.from_dataset_to_df_fn(data.train)
    # df = true_dataset_post_df[data.num_columns]
    # print('=====')
    # print(f'true_dataset_post_df: ')
    # print(df.head())
    # print(df.max())
    # print()
