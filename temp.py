from dp_data import load_domain_config, load_df
from utils import Dataset, Domain
from stats import ChainedStatistics, Marginals

import itertools
import numpy as np

import pdb
        
# load data
dataset_name = 'folktables_2018_coverage_CA'
root_path = 'dp-data-dev/datasets/preprocessed/folktables/1-Year/'

config = load_domain_config(dataset_name, root_path=root_path)
df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')

domain = Domain.fromdict(config)
data = Dataset(df_train, domain)

# define queries
k=2
bins = [2, 4, 8, 16, 32]

# run module
module = Marginals.get_all_kway_combinations(domain, k=k, bins=bins)
# module = Marginals.get_all_kway_mixed_combinations_v1(domain, k=k, bins=bins)
stat_module = ChainedStatistics([module])

stat_module.fit(data)
true_stats = stat_module.get_all_true_statistics()
stat_fn = stat_module.get_dataset_statistics_fn()

answers = stat_fn(data)
print(answers.shape[0])

# check
x = domain.config.copy()
for key, val in x.items():
    if val == 1:
        x[key] = sum(bins)
    
num_queries = itertools.combinations(x.values(), k)
num_queries = np.array(list(num_queries))
num_queries = num_queries.prod(-1)
num_queries = num_queries.sum()
print(num_queries)
