import jax
import jax.numpy as jnp
import optax
from models import Generator
from utils import Dataset
from stats import ChainedStatistics


# @dataclass
class RelaxedProjection(Generator):

    def __init__(self, domain, data_size, iterations=1000, learning_rate=0.001, stop_loss_time_window=20, print_progress=False):
        # super().__init__(domain, stat_module, data_size, seed)
        self.domain = domain
        self.data_size = data_size
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.early_stop_percent = 0.001
        self.stop_loss_time_window = stop_loss_time_window
        self.print_progress = print_progress

    def __str__(self):
        return f'RP(lr={self.learning_rate:.4f})'

    def fit(self, key, stat: ChainedStatistics, init_data: Dataset=None, tolerance=0, adaptive_epoch=1):

        data_dim = self.domain.get_dimension()

        # compute_loss = lambda params, sigmoid: optax.l2_loss(stat_fn(params['w'], sigmoid), true_stats)
        target_stats = stat.get_selected_noised_statistics()
        stat_fn = stat.get_selected_statistics_fn()
        softmax_fn = lambda X: Dataset.apply_softmax(self.domain, X)
        compute_loss = lambda params: jnp.linalg.norm(target_stats - stat_fn(softmax_fn(params['w']))) ** 2


        true_target_stats = stat.get_selected_statistics_without_noise()
        true_losses_fn = jax.jit(lambda params: jnp.abs(true_target_stats - stat_fn(softmax_fn(params['w']))))
        priv_losses_fn = jax.jit(lambda params: jnp.abs(target_stats - stat_fn(softmax_fn(params['w']))))


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
        best_loss = 100
        stop_loss_window = 20
        self.early_stop_init()
        last_loss = 100
        for t in range(self.iterations):
            loss = compute_loss(params)
            updates, self.opt_state = update_fn(params, self.opt_state)
            params = optax.apply_updates(params, updates)
            smooth_loss_sum += loss

            best_loss = min(best_loss, loss)

            if last_loss is None or loss < last_loss * 0.99 or t > self.iterations-2 :
                if self.print_progress:
                    priv_losses = priv_losses_fn(params)
                    losses = true_losses_fn(params)
                    print(f'epoch {t:<5}). Loss={float(loss):.6f}, '
                          f'\tpriv error(max/l2)={float(priv_losses.max()):.5f}/{float(jnp.linalg.norm(priv_losses, ord=2)):.6f}, '
                          f'\ttrue error(max/l2)={float(losses.max()):.5f}/{float(jnp.linalg.norm(losses, ord=2)):.6f}, '
                          # f'ave error ={float(jnp.linalg.norm(losses, ord=2)):.6f}'
                          # f'max error ={float(losses.max()):.5f}, '
                          )
                last_loss = loss


            if t > 50:
                if self.early_stop(t, best_loss):
                    if self.print_progress:
                        print(f'\tStop early at {t}')
                    break


        params['w'] = softmax_fn(params['w'])
        self.synthetic_data = params['w']

        self.key, subkey = jax.random.split(self.key, 2)
        X_onehot = Dataset.get_sample_onehot(subkey, self.domain, self.synthetic_data, num_samples=30)

        sync_dataset = Dataset.from_onehot_to_dataset(self.domain, X_onehot)
        return sync_dataset

        # return self.synthetic_data


    @staticmethod
    def train_help(domain, target_stats, stat_fn, init_data, learning_rate, print_progress):

        softmax_fn = lambda X: Dataset.apply_softmax(domain, X)
        compute_loss = lambda params: jnp.linalg.norm(target_stats - stat_fn(softmax_fn(params['w']))) ** 2

        optimizer = optax.adam(learning_rate)
        # Obtain the `opt_state` that contains statistics for the optimizer.
        params = {'w': init_data}

        opt_state = optimizer.init(params)
        update_fn = lambda params,  state: optimizer.update(jax.grad(compute_loss)(params), state)
        update_fn = jax.jit(update_fn)

        last_loss = None
        smooth_loss_sum = 0
        best_loss = 100
        stop_loss_window = 20
        # self.early_stop_init()
        last_loss = 100
        iterations = 5000
        for t in range(iterations):
            loss = compute_loss(params)
            updates, opt_state = update_fn(params, opt_state)
            params = optax.apply_updates(params, updates)
            smooth_loss_sum += loss

            best_loss = min(best_loss, loss)

            if last_loss is None or loss < last_loss * 0.99 or t > iterations-2 :
                if print_progress:
                    print(f'epoch {t:<5}). Loss={float(loss):.6f}, '
                          )
                last_loss = loss
            # if t > 50:
            #
            #     if print_progress:
            #         print(f'\tStop early at {t}')
            #     break

        params['w'] = softmax_fn(params['w'])
        return params

