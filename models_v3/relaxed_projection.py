import jax
import jax.numpy as jnp
import optax
from models_v3 import Generator
from utils import Dataset, Domain
from stats_v2 import Statistic
from dataclasses import dataclass
from stats_v3 import Marginals, PrivateMarginalsState


# @dataclass
class RelaxedProjection(Generator):
    # domain: Domain
    # data_size: int
    # iterations: int
    # learning_rate: float = 0.005
    # print_progress: bool = False
    # early_stop_percent: float = 0.001

    def __init__(self, domain, data_size,iterations=1000, learning_rate=0.001, print_progress=False):
        # super().__init__(domain, stat_module, data_size, seed)
        self.domain = domain
        self.data_size = data_size
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.early_stop_percent = 0.001
        self.print_progress = print_progress

    def __str__(self):
        return f'RP(lr={self.learning_rate:.4f})'
    def fit(self, key, stat: PrivateMarginalsState, init_X=None, tolerance=0):

        data_dim = self.domain.get_dimension()

        # stat_fn = jax.jit(stat_module.get_differentiable_stats_fn())



        # compute_loss = lambda params, sigmoid: optax.l2_loss(stat_fn(params['w'], sigmoid), true_stats)
        softmax_fn = lambda X: Dataset.apply_softmax(self.domain, X)
        compute_loss = lambda params: jnp.linalg.norm(stat.get_diff_stats(softmax_fn(params['w'])) - stat.get_priv_stats())**2

        self.key, subkey = jax.random.split(key, 2)
        self.synthetic_data = softmax_fn(jax.random.uniform(subkey, shape=(self.data_size, data_dim), minval=0, maxval=1))

        self.optimizer = optax.adam(self.learning_rate)
        # Obtain the `opt_state` that contains statistics for the optimizer.
        params = {'w': self.synthetic_data}
        self.opt_state = self.optimizer.init(params)

        update_fn = lambda params,  state: self.optimizer.update(jax.grad(compute_loss)(params), state)
        update_fn = jax.jit(update_fn)

        last_loss = None
        smooth_loss_sum = 0
        stop_loss_window = 20
        for t in range(self.iterations):
            # params['w'] = softmax_fn(params['w'])
            loss = compute_loss(params)
            updates, self.opt_state = update_fn(params, self.opt_state)
            params = optax.apply_updates(params, updates)
            # self.opt_state[0].mu['w'] = params['w']
            smooth_loss_sum += loss


            priv_max_error = stat.priv_diff_loss_inf(params['w'])
            if self.print_progress:
                print(f'epoch {t:<3}). Loss={loss}, priv_max_error ={priv_max_error}')
            if priv_max_error < tolerance:
                if self.print_progress:
                    print(f'Eary stop at {t}')
                break
            # Stop Early code
            if t >= stop_loss_window and t % stop_loss_window == 0:
                smooth_loss_avg = smooth_loss_sum / stop_loss_window
                if t > stop_loss_window:
                    loss_change = jnp.abs(smooth_loss_avg - last_loss) / last_loss
                    if loss_change < self.early_stop_percent:
                        break
                last_loss = smooth_loss_avg
                smooth_loss_sum = 0

        params['w'] = softmax_fn(params['w'])
        self.synthetic_data = params['w']

        self.key, subkey = jax.random.split(self.key, 2)
        X_onehot = Dataset.get_sample_onehot(subkey, self.domain, self.synthetic_data, num_samples=30)

        if self.print_progress:
            priv_max_error = stat.priv_diff_loss_inf(self.synthetic_data)
            print(f'Debug1: synthetic_data max error = priv_max_error ={priv_max_error}')
            priv_max_error = stat.priv_diff_loss_inf(X_onehot)
            print(f'Debug2: X_onehot max error = priv_max_error ={priv_max_error}')
        sync_dataset = Dataset.from_onehot_to_dataset(self.domain, X_onehot)
        return sync_dataset

        # return self.synthetic_data


