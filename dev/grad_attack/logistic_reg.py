import warnings

import jax.random

warnings.filterwarnings("ignore", category=UserWarning)

import jax.numpy as jnp
from jax import grad
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils.utils_data import Dataset, Domain
from stats import GeneralQuery, ChainedStatistics
import pandas as pd
from models import GSD_nondp


FS = (8, 4)  # figure size
RS = 124  # random seed

def logistic(r):
    return 1 / (1 + jnp.exp(-r))

def predict(c, w, X):
    return logistic(jnp.dot(X, w) + c)

def cost(c, w, X, y, eps=1e-14):
    lmbd = 0
    n = y.size
    p = predict(c, w, X)
    p = jnp.clip(p, eps, 1 - eps)  # bound the probabilities within (0,1) to avoid ln(0)
    return -jnp.sum(y * jnp.log(p) + (1 - y) * jnp.log(1 - p)) / n + 0.5 * lmbd * (
        jnp.dot(w, w) + c * c
    )


from sklearn.datasets import make_classification
import numpy as np

if __name__== '__main__':
    # Data parameters
    DATA_SIZE = 10000
    TRAIN_DATA = 100
    GRADS = 50
    d = 2
    seed = 0

    # LG parameters
    n_iter = 1000
    eta = 5e-2
    tol = 1e-6

    X, y = make_classification(n_samples=DATA_SIZE, n_features=d, n_informative=d, n_redundant=0,
                               n_repeated=0, random_state=RS)
    x_max = X.max(axis=0)
    x_min = X.min(axis=0)
    X = (X - x_min) / (x_max - x_min)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_DATA, stratify=y, random_state=RS)


    data_train_np = np.column_stack((X_train, y_train))
    cols = [f'f{i}' for i in range(d)] + ['label']
    domain = Domain(cols, [1 for _ in range(d)] + [2])
    df = pd.DataFrame(data_train_np, columns=cols)
    df.to_csv('real_train.csv')
    data = Dataset(df, domain)
    n_feat = X.shape[1]



    # Begin training
    c_0 = 1.0
    w_0 = 1.0e-5 * jnp.ones(n_feat)

    w = w_0
    c = c_0
    new_cost = float(cost(c, w, X_train, y_train))
    cost_hist = [new_cost]


    def create_grad_query(c_arg, w_arg):
        def grad_fn(X):
            X_temp = X[:, :-1]
            y_temp = X[:, -1]
            grad1 = grad(cost, argnums=0)(c_arg, w_arg, X_temp, y_temp)
            grad2 = grad(cost, argnums=1)(c_arg, w_arg, X_temp, y_temp)
            return jnp.concatenate((jnp.array(grad1).reshape(1), grad2))
        return grad_fn

    grad_queries = []
    grad_answers = []
    for i in range(n_iter):
        c_current = c

        if i % (n_iter // GRADS) == 0:
            grad_fn = create_grad_query(c_current, w)
            grad_queries.append(grad_fn)
            grad_answers.append(grad_fn(data_train_np))

        grad1 = grad(cost, argnums=0)(c_current, w, X_train, y_train)
        grad2 = grad(cost, argnums=1)(c_current, w, X_train, y_train)
        # grads = jnp.concatenate((jnp.array(grad1).reshape(1), grad2))

        c -= eta * grad1
        w -= eta * grad2

        new_cost = float(cost(c, w, X_train, y_train))
        cost_hist.append(new_cost)
        if (i > 20) and (i % 10 == 0):
            if jnp.abs(cost_hist[-1] - cost_hist[-20]) < tol:
                print(f"Exited loop at iteration {i}")
                break


    y_pred_proba = predict(c, w, X_test)
    y_pred = jnp.array(y_pred_proba)
    y_pred = jnp.where(y_pred < 0.5, y_pred, 1.0)
    y_pred = jnp.where(y_pred >= 0.5, y_pred, 0.0)
    print(classification_report(y_test, y_pred))


    # GSD

    @jax.jit
    def stat_fn(X):
        ans = [query(X) for query in grad_queries]
        return jnp.concatenate(ans)
    target_stats = jnp.concatenate(grad_answers)
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

    sync_data = algo.fit(key=jax.random.PRNGKey(RS),  statistics_fn=stat_fn, selected_statistics=target_stats)
    print(f'Saving ', f'sync_data_{GRADS}_{n_iter}.csv')
    sync_data.df.to_csv(f'sync_data_{GRADS}_{n_iter}.csv')

    grads_sync = stat_fn(sync_data.to_numpy())

    error = jnp.abs(grads_real - grads_sync)
    print(error.max())
