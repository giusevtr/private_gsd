

import pandas as pd
import itertools
from utils.utils_data import get_data


tasks = ['employment', 'coverage', 'income', 'mobility', 'travel']
# tasks = [ 'mobility', 'travel']
states = ['CA']

for task, state in itertools.product(tasks, states):
    data_name = f'folktables_2018_{task}_{state}'
    data = get_data(f'folktables_datasets/{data_name}-mixed-train',
                    domain_name=f'folktables_datasets/domain/{data_name}-mixed')

    print(task)
    print('cat:')
    for col in data.domain.get_categorical_cols():
        print(col, end=', ')
    print('\nnum:')
    for col in data.domain.get_numeric_cols():
        print(col, end=', ')
    print()
