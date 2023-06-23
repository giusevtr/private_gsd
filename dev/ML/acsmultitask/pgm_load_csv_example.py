import os
import pandas as pd
from dp_data import load_domain_config, load_df, DataPreprocessor, ml_eval
from utils import timer, Dataset, Domain


dataset_name = 'folktables_2018_multitask_CA'
root_path = '../../../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
config = load_domain_config(dataset_name, root_path=root_path)
df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')

domain = Domain.fromdict(config)
cat_cols = domain.get_categorical_cols()
num_cols = domain.get_numeric_cols()

print(f'train size: {df_train.shape}')
print(f'test size:  {df_test.shape}')
targets = ['JWMNP_bin', 'PINCP', 'ESR', 'MIG', 'PUBCOV']
features = []
for f in domain.attrs:
    if f not in targets:
        features.append(f)

model = 'LogisticRegression'
ml_fn = ml_eval.get_evaluate_ml(df_test, config, targets, models=[model])


datasets = [('folktables_2018_multitask_CA', '2_595_-1_hist_5_0', 'Hist'), # histogram
            ]

algo_name = 'PGM'
Res = []

for (dataset_name, query_name, query_name_short) in datasets:
    for epsilon in [1.0, 0.74, 0.52, 0.23, 0.07]:
        for alpha in [0.5]: # [0.5, 0.9]:
            for T in [25, 50, 75, 100]:
                for seed in range(3):
                    # path = f'./saved_syndata/{dataset_name}/{query_name}/{algo_name}/{epsilon}/adaptive/{alpha}/{T}/syndata_{seed}.csv'
                    path = f'./sync_data/PGM/{dataset_name}/{query_name}/{algo_name}/{epsilon}/adaptive/{alpha}/{T}/syndata_{seed}.csv'
                    print(path)
                    df_syndata = pd.read_csv(path)
                    res = ml_fn(df_syndata, seed=0)
                    res = res[res['Eval Data'] == 'Test']
                    # res = res[res['Metric'] == 'f1_macro']
                    print(algo_name + query_name, 'seed=', seed, 'eps=', epsilon)
                    print(res)
                    for i, row in res.iterrows():
                        # target = row['target']
                        # f1 = row['Score']
                        # Res.append(
                        #     [dataset_name, 'Yes', algo_name, query_name, T, 1, 'LR', target, epsilon, 'F1',
                        #      seed, f1])
                        target = row['target']
                        metric = row['Metric']
                        score = row['Score']
                        Res.append(
                            [dataset_name, 'Yes', algo_name, query_name_short, T, 1, model, target, epsilon,
                             metric, seed, score])


results = pd.DataFrame(Res, columns=['Data', 'Is DP',
                                     'Generator',
                                     'Statistics',
                                     'T', 'S',
                                     'Model',
                                     'Target',
                                     'epsilon', 'Metric', 'seed',
                                         'Score'])

os.makedirs('results_final', exist_ok=True)
file_path = f'results_final/results_pgm_{model}.csv'

print(f'Saving ', file_path)
results.to_csv(file_path, index=False)