import chex
import jax
import jax.numpy as jnp
import optax
from typing import Callable
from models import Generator
from dataclasses import dataclass
from utils import Dataset, Domain, timer
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
        self.stop_early = 50



    def __str__(self):
        return 'RP++'

    def fit(self, key, adaptive_statistic: AdaptiveStatisticState, init_data: Dataset=None, tolerance=0):

        softmax_fn = lambda X: Dataset.apply_softmax(self.domain, X)
        data_dim = self.domain.get_dimension()
        key, key2 = jax.random.split(key, 2)
        if adaptive_statistic.adaptive_rounds_count == 1:
            if self.print_progress: print('Initializing relaxed dataset')
            self.init_sync = softmax_fn(jax.random.uniform(key2, shape=(self.data_size, data_dim), minval=0, maxval=1))
        # self.init_sync = softmax_fn(jax.random.uniform(key2, shape=(self.data_size, data_dim), minval=0, maxval=1))

        # if init_data is None:
        #     init_sync = softmax_fn(jax.random.uniform(key2, shape=(self.data_size, data_dim), minval=0, maxval=1))
        # else:
        #     if init_data.df.shape[0] > self.data_size:
        #         init_data = init_data.sample(n=self.data_size, replace=False)
        #     init_sync = init_data.to_onehot()

        diff_stat_fn = adaptive_statistic.STAT_MODULE.get_diff_stat_fn(adaptive_statistic.get_statistics_ids())

        min_loss = None

        numeric_cols = self.domain.get_numeric_cols()
        num_idx = jnp.array([self.domain.get_attribute_onehot_indices(att) for att in numeric_cols]).squeeze()

        for lr in self.learning_rate:
            key, key2 = jax.random.split(key, 2)
            params = {'w': softmax_fn(self.init_sync.copy())}

            self.optimizer = optax.adam(lr)
            # self.optimizer.
            # Obtain the `opt_state` that contains statistics for the optimizer.
            opt_state = self.optimizer.init(params)

            target_stats = adaptive_statistic.get_private_statistics()
            # compute_loss = lambda params, sigmoid: jnp.linalg.norm(
            #     diff_stat_fn(softmax_fn(params['w']), sigmoid) - target_stats) ** 2
            def compute_loss(params, sigmoid):
                w = params['w']
                l1 = jnp.linalg.norm(diff_stat_fn(softmax_fn(w), sigmoid) - target_stats) ** 2
                l2 = jnp.sum(jax.nn.sigmoid(sigmoid * (w[:, num_idx] - 1)))
                l3 = jnp.sum(jax.nn.sigmoid(-sigmoid * (w[:, num_idx])))
                return l1 + l2 + l3
                # return l1

            # compute_loss_jit = (compute_loss)
            compute_loss_jit = jax.jit(compute_loss)
            update_fn = lambda pa, si, st: self.optimizer.update(jax.grad(compute_loss)(pa, si), st)
            update_fn_jit = jax.jit(update_fn)
            # update_fn_jit = (update_fn)

            params = self.fit_help(params, opt_state, compute_loss_jit, update_fn_jit, lr)
            sync = softmax_fn(params['w'])
            loss = jnp.linalg.norm(target_stats - diff_stat_fn(sync, 1024))
            if min_loss is None or loss < min_loss:
                self.init_sync = jnp.copy(sync)
                min_loss = loss

        # Dataset.from_onehot_to_dataset(self.domain, best_sync)
        key, key2 = jax.random.split(key, 2)

        oh = Dataset.get_sample_onehot(key2, self.domain, X_relaxed=self.init_sync, num_samples=20)
        return Dataset.from_onehot_to_dataset(self.domain, oh)

    def fit_help(self, params, opt_state, compute_loss_jit, update_fn_jit, lr):

        stop_early = self.stop_early

        self.early_stop_init()
        best_loss = compute_loss_jit(params, 1024)
        iters = 0
        t0 = timer()
        t1 = timer()
        loss_hist = []
        for i in range(11):
            # sigmoid = i ** 2
            sigmoid = 2 ** i
            if self.print_progress: print(f'sigmoid={sigmoid}:')
            for t in range(2000):
                iters += 1
                loss = compute_loss_jit(params, 1024)
                loss_hist.append(loss)
                updates, opt_state = update_fn_jit(params, sigmoid, opt_state)
                params = optax.apply_updates(params, updates)
                if (t > stop_early and len(loss_hist)> 2 * stop_early and loss >= 0.999 * loss_hist[-10]):
                    if self.print_progress:
                        t0 = timer(t0, f'Stop early at {t} for sigmoid = {sigmoid} current loss={loss:.5f}. time=')
                    break

                if self.print_progress:
                    total_loss = compute_loss_jit(params, 1024)
                    if total_loss < 0.95 * best_loss:
                        t1 = timer(t1, f't={t:<5} sigmoid={sigmoid:<5}| total_loss={total_loss:<8.5f}. time=')
                        best_loss = total_loss

        return params

