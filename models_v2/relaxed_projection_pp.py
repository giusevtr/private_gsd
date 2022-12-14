import jax
import jax.numpy as jnp
import optax
from models_v2 import Generator
from dataclasses import dataclass
from utils import Dataset, Domain
from stats_v2 import Statistic

@dataclass
class RelaxedProjectionPP(Generator):
    domain: Domain
    data_size: int
    iterations: int
    learning_rate: tuple = (0.001,)
    print_progress: bool = False
    early_stop_percent: float = 0.001

    def __str__(self):
        return 'RP++'

    def fit(self, key, true_stats: jnp.ndarray, stat_module: Statistic,  init_X=None):
        # self.optimizer = optax.adam(lr)
        stat_fn = stat_module.get_differentiable_stats_fn()
        compute_loss = lambda params, sigmoid: jnp.linalg.norm(stat_fn(params['w'], sigmoid) - true_stats)**2
        compute_loss_jit = jax.jit(compute_loss)
        update_fn = lambda pa, si, st: self.optimizer.update(jax.grad(compute_loss)(pa, si), st)
        update_fn_jit = jax.jit(update_fn)

        min_loss = None
        best_sync = None
        for lr in self.learning_rate:
            key, key2 = jax.random.split(key, 2)
            sync = self.fit_help(key2, compute_loss_jit, update_fn_jit, lr)
            loss = jnp.linalg.norm(true_stats - stat_fn(sync, 10000))
            if best_sync is None or loss < min_loss:
                best_sync = jnp.copy(sync)
                min_loss = loss

        # Dataset.from_onehot_to_dataset(self.domain, best_sync)
        return Dataset.from_onehot_to_dataset(self.domain, best_sync)

    def fit_help(self, key, compute_loss_jit, update_fn_jit, lr):
        data_dim = self.domain.get_dimension()
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

