import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier

QUANTILES = 50
seed = 0

# Create score function using the F1-macro metric
score_fn = make_scorer(f1_score, average='macro')

ordinal = ['Pregnant', 'Plasma', 'BloodPressure', 'Triceps', 'Age', 'Label']
columns = ['Pregnant', # 17
           'Plasma', # 199
           'BloodPressure', # 122
           'Triceps',# 99
           'Insulin', # 846
           'BMI', # 67.1
           'Diabetes_pedigree', # 2.342
           'Age', # 60
           'Label']


target = 'Label'
features = list(columns)
features.remove(target)

# Optimal parameters obtained from diabetes_ml_gridsearch.py
learning_rate = 0.1185499403580282
max_depth = 7
min_child_weight = 1
gamma = 0.16393189784441567
subsample = 0.8407621684817781
reg_lambda = 9.975971921758958

# Evaluate models on test data.
original_test_scores = []
synthetic_test_scores = []

for seed in range(5):
    data_path = '../data2/data'
    X_train = np.load(f'{data_path}/diabetes/kfolds/{seed}/X_num_train.npy')
    X_test = np.load( f'{data_path}/diabetes/kfolds/{seed}/X_num_test.npy')
    X_val = np.load(  f'{data_path}/diabetes/kfolds/{seed}/X_num_val.npy')
    y_train = np.load(f'{data_path}/diabetes/kfolds/{seed}/y_train.npy')
    y_test = np.load( f'{data_path}/diabetes/kfolds/{seed}/y_test.npy')
    y_val = np.load(  f'{data_path}/diabetes/kfolds/{seed}/y_val.npy')


    # Load synthetic data and extract features
    sync_df = pd.read_csv(f'sync_data/diabetes/2/inf/N/oneshot/sync_data_{seed}.csv')

    X_sync = sync_df[features]
    y_sync = sync_df[target]

    for rs in range(10):
        # Train a model on the original data and save test score
        model = XGBClassifier(
            # learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight,
            #                   gamma=gamma, subsample=subsample, reg_lambda=reg_lambda,
            random_state=rs
                              )

        model.fit(X_train, y_train)
        original_train = score_fn(model, X_train, y_train)
        original_val = score_fn(model, X_val, y_val)
        original_test = score_fn(model, X_test, y_test)
        original_test_scores.append(original_test)


        # Train a model on the synthetic data and save test score
        model_sync = XGBClassifier(
            # learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight,
            #                   gamma=gamma, subsample=subsample, reg_lambda=reg_lambda,
                        random_state=rs)
        model_sync.fit(X_sync, y_sync)
        synthetic_train = score_fn(model_sync, X_sync, y_sync)
        synthetic_val = score_fn(model_sync, X_val, y_val)
        synthetic_test = score_fn(model_sync, X_test, y_test)
        synthetic_test_scores.append(synthetic_test)

        print(f'Original: \t{original_train:.5f}\t{original_val:.5f}\t{original_test:.5f}')
        print(f'Synthetic:\t{synthetic_train:.5f}\t{synthetic_val:.5f}\t{synthetic_test:.5f}')
        print()

print()
print(f'Original average test score:  {np.mean(original_test_scores):.5f}, std={np.std(original_test_scores):.5f}')
print(f'Synthetic average test score: {np.mean(synthetic_test_scores):.5f}, std={np.std(synthetic_test_scores):.5f}')
