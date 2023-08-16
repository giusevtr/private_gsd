import jax.numpy as jnp
from experiments.run_exp import run_experiment, get_ml_score_fn

def get_constraint_fn(domain):
    # Build  constraint
    race_feat = 'cat_5'

    sex_feat = 'cat_6'
    sex_att_ind = domain.get_attribute_indices([sex_feat])[0]
    label_ind = domain.get_attribute_indices(['Label'])[0]
    def constraint_fn(X):
        males = (X[:, sex_att_ind] == 1).mean()
        loss1 = jnp.abs(males - 0.5)**2

        males_inc = ((X[:, sex_att_ind] == 1) & (X[:, label_ind] == 1)).mean()
        females_inc = ((X[:, sex_att_ind] == 0) & (X[:, label_ind] == 1)).mean()
        loss2 = (males_inc - females_inc)**2
        return loss1 + loss2

SEEDS = [0, 1, 2, 3, 4]

datasets = [
    ('miniboone', '2000'),
    ('adult', '2000'),
    ('diabetes', 'N'),
    ('churn2', '2000'),
    ('cardio', '2000'),
    ('higgs-small', '2000'),
    ('wilt', '2000'),
]
for data_name, data_size in datasets:

    run_experiment(data_name=data_name, data_size_str=data_size,  seeds=SEEDS, verbose=False)



