from stats import PrefixDiff, Prefix
import jax

from utils import Domain, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


domain = Domain(['A', 'B'], [10, 1])


key = jax.random.PRNGKey(0)
prefix = Prefix.get_kway_prefixes(domain, k_cat=0, k_num=1, random_prefixes=1, rng=key)
prefix_idff = PrefixDiff.get_kway_prefixes(domain, k_cat=0, k_num=1, random_prefixes=1, rng=key)
prefix_fn = prefix._get_dataset_statistics_fn(jitted=True)
prefix_diff_fn = prefix_idff._get_dataset_statistics_fn(jitted=True)


pre = []
pre_diff = []
for v in np.linspace(0, 1, 100):
    df = pd.DataFrame([[0, v]], columns=['A', 'B'])
    data = Dataset(df, domain)
    ans = prefix_fn(data)
    ans_diff = prefix_diff_fn(data)
    print(f'v={v:.4f}', 'prefix_fn = ',ans, 'diff=', ans_diff)
    pre.append(ans)
    pre_diff.append(ans_diff)


plt.plot(pre, label='halfspaces')
plt.plot(pre_diff, label='prefix_diff')
plt.legend()
plt.show()
