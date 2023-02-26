import chex
import jax
import jax.numpy as jnp
import optax
from typing import Callable
from models import Generator
from dataclasses import dataclass
from utils import Dataset, Domain, timer
from stats import Marginals, AdaptiveStatisticState, ChainedStatistics

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
        self.stop_early = 100
        self.iterations = 10000



    def __str__(self):
        return 'RAP++'

    def fit(self, key, adaptive_statistic: ChainedStatistics, init_data: Dataset=None, tolerance=0, adaptive_epoch=1):

        softmax_fn = lambda X: Dataset.apply_softmax(self.domain, X)
        data_dim = self.domain.get_dimension()
        key, key2 = jax.random.split(key, 2)

        # Check if this is the first adaptive round. If so, then initialize a synthetic data
        # selected_workloads = len(adaptive_statistic.selected_workloads)
        # if adaptive_epoch == 1:
        if self.print_progress: print('Initializing relaxed dataset')
        self.init_sync = softmax_fn(jax.random.uniform(key2, shape=(self.data_size, data_dim), minval=0, maxval=1))

        target_stats = adaptive_statistic.get_selected_noised_statistics()
        diff_stat_fn = adaptive_statistic.get_selected_statistics_fn()

        min_loss = None

        numeric_cols = self.domain.get_numeric_cols()
        num_idx = jnp.array([self.domain.get_attribute_onehot_indices(att) for att in numeric_cols]).squeeze()

        for lr in self.learning_rate:
            key, key2 = jax.random.split(key, 2)
            params = {'w': softmax_fn(self.init_sync.copy())}

            self.optimizer = optax.adam(lr)
            # Obtain the `opt_state` that contains statistics for the optimizer.
            opt_state = self.optimizer.init(params)

            def compute_loss(params, sigmoid):
                w = params['w']
                # Distance to the target statistics
                loss = jnp.linalg.norm(diff_stat_fn(softmax_fn(w), sigmoid=sigmoid) - target_stats) ** 2
                # Add a penalty if any numeric features moves outsize the range [0,1]
                loss += jnp.sum(jax.nn.sigmoid(2**15 * (w[:, num_idx] - 1)))
                loss += jnp.sum(jax.nn.sigmoid(-2**15 * (w[:, num_idx])))
                return loss
            # def debug_compute_loss(params):
            #     w = params['w']
            #     # Distance to the target statistics
            #     max_error = jnp.abs(diff_stat_fn(softmax_fn(w)) - target_stats).max()
            #     return max_error
            compute_loss_jit = jax.jit(compute_loss)
            update_fn = lambda pa, si, st: self.optimizer.update(jax.grad(compute_loss)(pa, si), st)
            update_fn_jit = jax.jit(update_fn)
            # update_fn_jit = (update_fn)

            params = self.fit_help(params, opt_state, compute_loss_jit, update_fn_jit)
            sync = softmax_fn(params['w'])
            loss = jnp.linalg.norm(target_stats - diff_stat_fn(sync))
            if min_loss is None or loss < min_loss:
                self.init_sync = jnp.copy(sync)
                min_loss = loss

        # Dataset.from_onehot_to_dataset(self.domain, best_sync)
        key, key2 = jax.random.split(key, 2)

        oh = Dataset.get_sample_onehot(key2, self.domain, X_relaxed=self.init_sync, num_samples=20)
        return Dataset.from_onehot_to_dataset(self.domain, oh)

    def fit_help(self, params, opt_state, compute_loss_jit, update_fn_jit):

        stop_early = self.stop_early

        self.early_stop_init()
        best_loss = compute_loss_jit(params, 2048)
        iters = 0
        t0 = timer()
        t1 = timer()
        loss_hist = []
        for i in range(15):
            # sigmoid = i ** 2
            sigmoid = 2 ** i
            if self.print_progress: print(f'sigmoid={sigmoid}:')
            for t in range(self.iterations):
                iters += 1
                loss = compute_loss_jit(params, 2048)
                # debug_loss = debug_compute_loss(params)
                loss_hist.append(loss)
                updates, opt_state = update_fn_jit(params, sigmoid, opt_state)
                params = optax.apply_updates(params, updates)
                if (t > stop_early and len(loss_hist)> 2 * stop_early and loss >= 0.999 * loss_hist[-10]):
                    if self.print_progress:
                        t0 = timer(t0, f'Stop early at {t} for sigmoid = {sigmoid} current loss={loss:.5f}. time=')
                    break

                if self.print_progress:
                    total_loss = compute_loss_jit(params, 2048)
                    if total_loss < 0.95 * best_loss:
                        t1 = timer(t1, f't={t:<5} sigmoid={sigmoid:<5}| total_loss={total_loss:<8.5f}. time=')
                        best_loss = total_loss

        return params
