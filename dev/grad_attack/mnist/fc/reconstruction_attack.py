import pickle
import numpy as np
from dev.grad_attack.mnist.fc.train import num_epochs, batch_size, train_loader, get_grads
from models import GSD_nondp
import jax
import jax.numpy as jnp
from utils.utils_data import Dataset, Domain
import pandas as pd


d = 28 * 28
X, target = next(iter(train_loader))
X = X.reshape(-1, d)

data_train_np = np.column_stack((X, target))

# data_train_np = np.column_stack((X_train, y_train))
cols = [f'f{i}' for i in range(d)] + ['label']
domain = Domain(cols, [1 for _ in range(d)] + [10])
df = pd.DataFrame(data_train_np, columns=cols)
df.to_csv('real_train.csv')
data = Dataset(df, domain)
n_feat = X.shape[1]


def create_grad_query(params):
    def grad_fn(X):
        X_temp = X[:, :-1]
        y_temp = X[:, -1]
        grad, value = get_grads(params, X_temp, y_temp)
        grads_flat = jnp.concatenate([jnp.concatenate((g[0], g[1])) for g in grad])
        return grads_flat
    return grad_fn


queries = []
answers = []
for i in range(num_epochs):
    params_file = f'params/param_{i}.p'
    actual_grad_file = f'params/grad_param_{i}.p'
    param = pickle.load(open(params_file, "rb"))
    targets_grad = pickle.load(open(actual_grad_file, "rb"))

    grad_query = create_grad_query(param)
    queries.append(grad_query)
    answers.append(targets_grad)


# GSD
@jax.jit
def stat_fn(X):
    ans = [query(X) for query in queries]
    return jnp.concatenate(ans)


target_stats = jnp.concatenate(answers)
grads_real = stat_fn(data.to_numpy())

error = jnp.abs(grads_real - target_stats)
print('debug: ', error.max())


# Choose Private-GSD parameters
algo = GSD_nondp(domain=data.domain,
                 print_progress=True,
                 stop_early=True,
                 num_generations=20000,
                 population_size_muta=50,
                 population_size_cross=50,
                 data_size=100)

sync_data = algo.fit(key=jax.random.PRNGKey(0),  statistics_fn=stat_fn, selected_statistics=target_stats)
print(f'Saving ', f'sync_data_{num_epochs}_{batch_size}.csv')
sync_data.df.to_csv(f'sync_data_{num_epochs}_{batch_size}.csv')

grads_sync = stat_fn(sync_data.to_numpy())

error = jnp.abs(grads_real - grads_sync)
print(error.max())
