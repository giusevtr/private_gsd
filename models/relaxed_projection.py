import jax
import jax.numpy as jnp
import optax
from models import Generator
from utils import Dataset
from stats import Statistic


class RelaxedProjection(Generator):
    def __init__(self, domain, stat_module: Statistic, data_size, seed, iterations=1000, learning_rate=0.001):
        super().__init__(domain, stat_module, data_size, seed)
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.early_stop_percent = 0.001

    def __str__(self):
        return f'RP(lr={self.learning_rate:.4f})'

    @staticmethod
    def get_generator(iterations=1000, learning_rate=0.001):
        generator_init = lambda domain, stat_moudule, data_size, seed:   RelaxedProjection(domain, stat_moudule,
                                                                                           data_size, seed,
                                                                               iterations, learning_rate)
        return generator_init

    def fit(self, true_stats, init_X=None):

        stat_fn = jax.jit(self.stat_module.get_differentiable_stats_fn())

        self.key, subkey = jax.random.split(self.key, 2)
        self.synthetic_data = jax.random.uniform(subkey, shape=(self.data_size, self.data_dim), minval=0, maxval=1)

        self.optimizer = optax.adam(self.learning_rate)
        # Obtain the `opt_state` that contains statistics for the optimizer.
        params = {'w': self.synthetic_data}
        self.opt_state = self.optimizer.init(params)
        # compute_loss = lambda params, sigmoid: optax.l2_loss(stat_fn(params['w'], sigmoid), true_stats)
        softmax_fn = lambda X: Dataset.apply_softmax(self.domain, X)
        compute_loss = lambda params: jnp.linalg.norm(stat_fn(softmax_fn(params['w'])) - true_stats)**2

        compute_loss_jit = jax.jit(compute_loss)

        update_fn = lambda params,  state: self.optimizer.update(jax.grad(compute_loss)(params), state)
        update_fn_jit = jax.jit(update_fn)


        last_loss = None
        smooth_loss_sum = 0
        stop_loss_window = 20
        for t in range(self.iterations):
            loss = compute_loss_jit(params)
            updates, self.opt_state = update_fn_jit(params, self.opt_state)
            params = optax.apply_updates(params, updates)
            smooth_loss_sum += loss

            # Stop Early code
            if t >= stop_loss_window and t % stop_loss_window == 0:
                smooth_loss_avg = smooth_loss_sum / stop_loss_window
                if t > stop_loss_window:
                    loss_change = jnp.abs(smooth_loss_avg - last_loss) / last_loss
                    if loss_change < self.early_stop_percent:
                        break
                # print(f'smooth_loss_avg={smooth_loss_avg}')
                last_loss = smooth_loss_avg
                smooth_loss_sum = 0

        self.synthetic_data = softmax_fn(params['w'])

        self.key, subkey = jax.random.split(self.key, 2)
        X_onehot = Dataset.get_sample_onehot(subkey, self.domain, self.synthetic_data, num_samples=30)

        sync_dataset = Dataset.from_onehot_to_dataset(self.domain, X_onehot)
        return sync_dataset

        # return self.synthetic_data


