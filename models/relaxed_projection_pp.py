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

    def __init__(self, domain, data_size, iterations=1000, learning_rate=(0.001,), stop_loss_time_window=20, print_progress=False):
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

    def fit(self, key, adaptive_statistic: AdaptiveStatisticState, init_sync=None, tolerance=0):
        # self.optimizer = optax.adam(lr)
        # stat_fn = stat.get_differentiable_stats_fn()



        update_functions = []
        for stat_id in adaptive_statistic.statistics_ids:
            if stat_id not in self.CACHE:
                # elite_population_fn = adaptive_statistic.STAT_MODULE.get_marginals_fn[stat_id]()
                # mate_and_mute_rows_fn = adaptive_statistic.STAT_MODULE.get_marginals_fn[stat_id]()
                # mute_only_rows_fn = adaptive_statistic.STAT_MODULE.get_marginals_fn[stat_id]()
                # self.CACHE[stat_id] = ((jax.vmap(elite_population_fn, in_axes=(0, ))),
                #                        jax.jit(jax.vmap(mate_and_mute_rows_fn, in_axes=(0, ))),
                #                        jax.jit(jax.vmap(mute_only_rows_fn, in_axes=(0, )))
                #                        )

                diff_stat_fn = adaptive_statistic.STAT_MODULE.get_differentiable_fn[stat_id]
                target_stats = adaptive_statistic.STAT_MODULE.true_stats[stat_id]
                compute_loss = lambda params, sigmoid: jnp.linalg.norm(
                    diff_stat_fn(params['w'], sigmoid) - target_stats) ** 2

                update_fn = lambda pa, si, st: self.optimizer.update(jax.grad(compute_loss)(pa, si), st)
                update_fn_jit = jax.jit(update_fn)

                self.CACHE[stat_id] = update_fn_jit

        def update_fn(params, sigmoid, state):
            pass

        train_diff_fn: Callable

        true_stats = adaptive_statistic.get_true_statistics()
        # stat_fn = stat.private_diff_statistics_fn_jit
        # target_stats = stat_module.get_private_stats()
        target_stats = adaptive_statistic.get_private_statistics()

        compute_loss = lambda params, sigmoid: jnp.linalg.norm(train_diff_fn(params['w'], sigmoid) - target_stats)**2
        compute_loss_jit = jax.jit(compute_loss)
        update_fn = lambda pa, si, st: self.optimizer.update(jax.grad(compute_loss)(pa, si), st)
        update_fn_jit = jax.jit(update_fn)

        min_loss = None
        best_sync = None
        for lr in self.learning_rate:
            key, key2 = jax.random.split(key, 2)
            sync = self.fit_help(key2, stat.STAT_MODULE.domain, compute_loss_jit, update_fn_jit, lr)
            loss = jnp.linalg.norm(true_stats - stat_fn(sync, 10000))
            if best_sync is None or loss < min_loss:
                best_sync = jnp.copy(sync)
                min_loss = loss

        # Dataset.from_onehot_to_dataset(self.domain, best_sync)
        return Dataset.from_onehot_to_dataset(stat.STAT_MODULE.domain, best_sync)

    def fit_help(self, key, domain: Domain, compute_loss_jit, update_fn_jit, lr):
        data_dim = domain.get_dimension()
        rng, subkey = jax.random.split(key, 2)
        synthetic_data = jax.random.uniform(subkey, shape=(self.data_size, data_dim), minval=0, maxval=1)

        self.optimizer = optax.adam(lr)
        # Obtain the `opt_state` that contains statistics for the optimizer.
        params = {'w': synthetic_data}
        opt_state = self.optimizer.init(params)


        for sigmoid in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            last_loss = None
            smooth_loss_sum = 0
            stop_loss_window = 20
            for t in range(self.iterations):
                loss = compute_loss_jit(params, sigmoid)
                updates, opt_state = update_fn_jit(params, sigmoid, opt_state)
                params = optax.apply_updates(params, updates)
                smooth_loss_sum += loss

                # Stop Early code
                if t >= stop_loss_window and t % stop_loss_window == 0:
                    smooth_loss_avg = smooth_loss_sum / stop_loss_window
                    if t > stop_loss_window:
                        loss_change = jnp.abs(smooth_loss_avg - last_loss) / last_loss
                        if self.print_progress:
                            print(f'sigmoid {sigmoid:<3}, round {t:<3}: loss = ', loss, 'loss_change=', loss_change)
                        if loss_change < self.early_stop_percent:
                            break
                    last_loss = smooth_loss_avg
                    smooth_loss_sum = 0

        synthetic_data = params['w']
        return synthetic_data

