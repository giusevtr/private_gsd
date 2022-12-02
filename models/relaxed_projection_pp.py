import jax
import jax.numpy as jnp
import optax
from models import Generator


class RelaxedProjectionPP(Generator):
    def __init__(self, domain, stat_module, data_size, seed, iterations=1000, learning_rate=(0.001, ), print_progress=False):
        super().__init__(domain, stat_module, data_size, seed)
        self.iterations = iterations
        self.learning_rate = list(learning_rate)
        self.early_stop_percent = 0.001
        self.print_progress = print_progress

    def __str__(self):
        return 'RP++'

    @staticmethod
    def get_generator(iterations=1000, learning_rate=(0.001,), print_progress=False):
        generator_init = lambda data_dim, stat_module, data_size, seed:   RelaxedProjectionPP(data_dim,
                                                                                              stat_module,
                                                                                              data_size, seed,
                                                                                 iterations, learning_rate,
                                                                                 print_progress=print_progress)
        return generator_init

    def fit(self, true_stats,  init_X=None):

        stat_fn =  self.stat_module.get_differentiable_stats_fn()
        compute_loss = lambda params, sigmoid: jnp.linalg.norm(stat_fn(params['w'], sigmoid) - true_stats)**2
        compute_loss_jit = jax.jit(compute_loss)
        update_fn = lambda pa, si, st: self.optimizer.update(jax.grad(compute_loss)(pa, si), st)
        update_fn_jit = jax.jit(update_fn)

        min_loss = None
        best_sync = None
        for lr in self.learning_rate:
            sync = self.fit_help(compute_loss_jit, update_fn_jit, lr)
            loss = jnp.linalg.norm(true_stats - stat_fn(sync, 10000))
            if best_sync is None or loss < min_loss:
                best_sync = jnp.copy(sync)
                min_loss = loss

        return best_sync

    def fit_help(self, compute_loss_jit, update_fn_jit, lr):

        self.rng, subkey = jax.random.split(self.key, 2)
        self.synthetic_data = jax.random.uniform(subkey, shape=(self.data_size, self.data_dim), minval=0, maxval=1)

        self.optimizer = optax.adam(lr)
        # Obtain the `opt_state` that contains statistics for the optimizer.
        params = {'w': self.synthetic_data}
        self.opt_state = self.optimizer.init(params)


        for sigmoid in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            last_loss = None
            smooth_loss_sum = 0
            stop_loss_window = 20
            for t in range(self.iterations):
                loss = compute_loss_jit(params, sigmoid)
                updates, self.opt_state = update_fn_jit(params, sigmoid, self.opt_state)
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

        self.synthetic_data = params['w']
        return self.synthetic_data

