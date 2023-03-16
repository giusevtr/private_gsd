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
        self.stop_early = 50



    def __str__(self):
        return 'RAP++'

    def fit(self, key, adaptive_statistic: ChainedStatistics, init_data: Dataset=None, tolerance=0, adaptive_epoch=1):

        softmax_fn = lambda X: Dataset.apply_softmax(self.domain, X)
        softmax_fn = jax.jit(softmax_fn)
        data_dim = self.domain.get_dimension()
        key, key2 = jax.random.split(key, 2)

        # Check if this is the first adaptive round. If so, then initialize a synthetic data
        # selected_workloads = len(adaptive_statistic.selected_workloads)
        if adaptive_epoch == 1:
            if self.print_progress: print('Initializing relaxed dataset')
            self.init_params = jax.random.uniform(key2, shape=(self.data_size, data_dim), minval=0, maxval=1)

        target_stats = adaptive_statistic.get_selected_noised_statistics()
        diff_stat_fn = adaptive_statistic.get_selected_statistics_fn()

        numeric_cols = self.domain.get_numeric_cols()
        num_idx = jnp.array([self.domain.get_attribute_onehot_indices(att) for att in numeric_cols]).reshape(-1)

        @jax.jit
        def clip_numeric(D):
            D = D.at[:, num_idx].set(jnp.clip(D[:, num_idx], 0, 1))
            return D


        min_loss = None
        best_params = None
        for lr in self.learning_rate:
            key, key2 = jax.random.split(key, 2)
            params = {'w': self.init_params.copy()}

            self.optimizer = optax.adam(lr)
            # Obtain the `opt_state` that contains statistics for the optimizer.
            opt_state = self.optimizer.init(params)

            def compute_loss(params, sigmoid):
                w = params['w']
                sync = softmax_fn(w)
                # Distance to the target statistics
                loss = jnp.linalg.norm(diff_stat_fn(sync, sigmoid=sigmoid) - target_stats) ** 2
                return loss
            compute_loss_jit = jax.jit(compute_loss)
            update_fn = lambda pa, si, st: self.optimizer.update(jax.grad(compute_loss_jit)(pa, si), st)
            update_fn_jit = jax.jit(update_fn)
            params = self.fit_help(params, opt_state, compute_loss_jit, update_fn_jit, clip_numeric)
            loss = jnp.linalg.norm(target_stats - diff_stat_fn(softmax_fn(params['w'])))
            if min_loss is None or loss < min_loss:
                best_params = jnp.copy(params['w'])
                min_loss = loss

        self.init_params = jnp.copy(best_params)
        sync_softmax = softmax_fn(best_params)
        # Dataset.from_onehot_to_dataset(self.domain, best_sync)
        key, key2 = jax.random.split(key, 2)
        oh = Dataset.get_sample_onehot(key2, self.domain, X_relaxed=sync_softmax, num_samples=20)
        return Dataset.from_onehot_to_dataset(self.domain, oh)

    def fit_help(self, params, opt_state, compute_loss_jit, update_fn_jit, post_fn):

        stop_early = self.stop_early

        self.early_stop_init()
        best_loss = compute_loss_jit(params, 2**15)
        best_loss_last = best_loss
        iters = 0
        t0 = timer()
        t1 = timer()
        # TODO: Save best param
        best_params = None
        for i in range(16):
            # sigmoid = i ** 2
            sigmoid = 2 ** i

            t_sigmoid = timer()
            temp_params = params.copy()
            if self.print_progress: print(f'Sigmoid={sigmoid}: Starting loss={compute_loss_jit(temp_params, 2**15):.3f}')
            loss_hist = []
            sigmoid_last_loss = best_loss
            for t in range(self.iterations):
                iters += 1
                loss = compute_loss_jit(temp_params, 2**15)
                # debug_loss = debug_compute_loss(params)
                loss_hist.append(loss)
                updates, opt_state = update_fn_jit(temp_params, sigmoid, opt_state)
                temp_params = optax.apply_updates(temp_params, updates)
                temp_params['w'] = post_fn(temp_params['w'])
                if loss > 2 * best_loss:
                    if self.print_progress: print(f'\t3) Stop early at {t}. Final loss={loss:.5f}')
                    break
                if (t > stop_early + 1 and len(loss_hist) > 2 * stop_early and loss >= 0.999 * loss_hist[-self.stop_early]):
                    if self.print_progress:
                        t0 = timer(t0, f'\t3) Stop early at {t} for Sigmoid = {sigmoid}. Final loss={loss:.5f} time=')
                    break

                if self.print_progress:
                    total_loss = compute_loss_jit(params, 2**15)
                    if total_loss < 0.95 * sigmoid_last_loss:
                        t1 = timer(t1, f'\t2) t={t:<5} sigmoid={sigmoid:<5}| total_loss={total_loss:<8.5f}, best_loss={best_loss:8.5f}. time=')
                        sigmoid_last_loss = total_loss
            # update params:
            this_loss = compute_loss_jit(temp_params, 2**15)
            if self.print_progress:
                timer(t_sigmoid, f'\t\t4) End training| temp_params.total_loss={this_loss:<8.5f}, best_loss={best_loss:8.5f}. time=')
            if this_loss < best_loss:
                if self.print_progress:
                    print(f'\t\t5) Updating parameters...')
                best_loss = this_loss
                params = temp_params.copy()

        return params
