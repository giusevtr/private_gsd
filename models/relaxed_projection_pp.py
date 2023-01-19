import chex
import jax
import jax.numpy as jnp
import optax
from typing import Callable
from models import Generator
from dataclasses import dataclass
from utils import Dataset, Domain
from stats import Marginals, AdaptiveStatisticState

@dataclass
class RelaxedProjectionPP(Generator):
    # domain: Domain
    data_size: int
    iterations: int
    learning_rate: tuple = (0.001,)
    print_progress: bool = False
    early_stop_percent: float = 0.001

    def __init__(self, domain, data_size, iterations=1000, learning_rate=(0.001,),
                 stop_loss_time_window=20, print_progress=False):
        # super().__init__(domain, stat_module, data_size, seed)
        self.domain = domain
        self.data_size = data_size
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.early_stop_percent = 0.001
        self.stop_loss_time_window = stop_loss_time_window
        self.print_progress = print_progress
        self.CACHE = {}

    def __str__(self):
        return 'RP++'

    def fit(self, key, adaptive_statistic: AdaptiveStatisticState, init_data: Dataset=None, tolerance=0):

        softmax_fn = jax.jit(lambda X: Dataset.apply_softmax(self.domain, X))
        data_dim = self.domain.get_dimension()
        key, key2 = jax.random.split(key, 2)
        if init_data is None:
            init_sync = softmax_fn(jax.random.uniform(key2, shape=(self.data_size, data_dim), minval=0, maxval=1))
        else:
            init_sync = init_data.to_onehot()

        diff_stat_fn_list = []
        for stat_id in adaptive_statistic.statistics_ids:
            diff_data_fn = adaptive_statistic.STAT_MODULE.get_diff_stat_fn(stat_id)
            diff_stat_fn_list.append(diff_data_fn)
        def stat_fn(X, sigmoid):
            return jnp.concatenate([diff_stat(X, sigmoid) for diff_stat in diff_stat_fn_list])

        # stat_fn = adaptive_statistic.private_diff_statistics_fn_jit
        true_stats = adaptive_statistic.get_private_statistics()
        # compute_real_loss = lambda params: jnp.linalg.norm(adaptive_statistic.private_statistics_fn(softmax_fn(params['w'])) - true_stats)**2
        # compute_real_loss_jit = jax.jit(compute_real_loss)

        compute_loss = lambda params, sigmoid: jnp.linalg.norm(stat_fn(softmax_fn(params['w']), sigmoid) - true_stats)**2
        compute_loss_jit = jax.jit(compute_loss)
        update_fn = lambda pa, si, st: self.optimizer.update(jax.grad(compute_loss)(pa, si), st)
        update_fn_jit = jax.jit(update_fn)

        min_loss = None
        best_sync = None
        for lr in self.learning_rate:
            key, key2 = jax.random.split(key, 2)
            params = self.fit_help(compute_loss_jit, update_fn_jit, init_sync.copy(), lr)
            sync = softmax_fn(params['w'])
            loss = jnp.linalg.norm(true_stats - adaptive_statistic.private_statistics_fn_jit(sync))
            if best_sync is None or loss < min_loss:
                best_sync = jnp.copy(sync)
                min_loss = loss

        # Dataset.from_onehot_to_dataset(self.domain, best_sync)
        key, key2 = jax.random.split(key, 2)

        oh = Dataset.get_sample_onehot(key2, self.domain, X_relaxed=best_sync, num_samples=20)
        return Dataset.from_onehot_to_dataset(self.domain, oh)

    def fit_help(self, compute_loss_jit, update_fn_jit, init_sync, lr):
        softmax_fn = jax.jit(lambda X: Dataset.apply_softmax(self.domain, X))

        # data_dim = domain.get_dimension()
        # rng, subkey = jax.random.split(key, 2)

        self.optimizer = optax.adam(lr)
        # Obtain the `opt_state` that contains statistics for the optimizer.
        params = {'w': softmax_fn(init_sync)}
        opt_state = self.optimizer.init(params)


        best_loss = 1000
        self.early_stop_init()
        for t in range(self.iterations):
            for sigmoid in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                loss = compute_loss_jit(params, 1024)
                updates, opt_state = update_fn_jit(params, sigmoid, opt_state)
                params = optax.apply_updates(params, updates)

                if self.print_progress:
                    if loss < best_loss * 0.9:
                        print(f'sigmoid {sigmoid:<5}, round {t:<3}: loss = {loss:.6f}')
                        best_loss = loss

                if loss < 1e-4 or (t > 30 and self.early_stop(t, loss)):
                    if self.print_progress:
                        print(f'\tEary stop at {t}. loss={loss:.5}')
                    t = self.iterations
                    break
        return params

