import jax.numpy as jnp
from experiments.run_exp import run_experiment

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

SEEDS = [0, 1, 2]

datasets = [ 'adult', 'churn2',
              'cardio',
             'higgs-small',
             'wilt',
             'miniboone'
             ]
for data_name in datasets:
    if data_name == 'miniboone':
        run_experiment(data_name=data_name, data_size_str='N', k=2, max_marginal_size=1000, min_density=0.99, seeds=SEEDS)
    else:
        run_experiment(data_name=data_name, data_size_str='N', k=2, max_marginal_size=5000, min_density=0.99, seeds=SEEDS)
        run_experiment(data_name=data_name, data_size_str='N', k=2, max_marginal_size=50000, min_density=0.99, seeds=SEEDS)



