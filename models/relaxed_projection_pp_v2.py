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
class RelaxedProjectionPP_v2(Generator):
    # domain: Domain
    data_size: int
    iterations: int
    learning_rate: list
    print_progress: bool = False
    early_stop_percent: float = 0.001

    def __init__(self, domain, data_size, iterations=1000,
                 stop_loss_time_window=20, print_progress=False):
        # super().__init__(domain, stat_module, data_size, seed)
        self.domain = domain
        self.data_size = data_size
        self.iterations = iterations
        self.early_stop_percent = 0.001
        self.stop_loss_time_window = stop_loss_time_window
        self.print_progress = print_progress
        self.CACHE = {}
        self.stop_early = 10
        self.init_sync_data = None

        self.learning_rate = [0.00005, 0.00007, 0.0001, 0.0002, 0.0003, 0.0004,
                   0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003]



    def __str__(self):
        return 'RAP++'

    def fit(self, key, adaptive_statistic: ChainedStatistics, init_data: Dataset=None, tolerance=0, adaptive_epoch=1):

        softmax_fn = lambda X: Dataset.apply_softmax(self.domain, X)
        softmax_fn = jax.jit(softmax_fn)
        data_dim = self.domain.get_dimension()
        key, key2 = jax.random.split(key, 2)

        # Check if this is the first adaptive round. If so, then initialize a synthetic data
        # selected_workloads = len(adaptive_statistic.selected_workloads)
        start_params = []
        init_params_1 = {'w': softmax_fn(jax.random.uniform(key2, shape=(self.data_size, data_dim), minval=0, maxval=1))}
        start_params.append(init_params_1)
        if adaptive_epoch > 1:
            # if self.print_progress: print('Initializing relaxed dataset')
            start_params.append({'w': self.init_sync_data})
            # self.init_params = jax.random.uniform(key2, shape=(self.data_size, data_dim), minval=0, maxval=1)
        # params = {'w': self.init_params.copy()}

        target_stats = adaptive_statistic.get_selected_noised_statistics()
        diff_stat_fn = adaptive_statistic.get_selected_statistics_fn()

        numeric_cols = self.domain.get_numeric_cols()
        num_idx = jnp.array([self.domain.get_attribute_onehot_indices(att) for att in numeric_cols]).reshape(-1)

        @jax.jit
        def clip_numeric(D):
            D = D.at[:, num_idx].set(jnp.clip(D[:, num_idx], 0, 1))
            return D



        # self.optimizer = optax.adam(lr)
        # Obtain the `opt_state` that contains statistics for the optimizer.
        # opt_state = self.optimizer.init(params)
        self.optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=0)

        def compute_loss(params, sigmoid):
            w = params['w']
            sync = softmax_fn(w)
            loss = jnp.linalg.norm(diff_stat_fn(sync, sigmoid=sigmoid) - target_stats) ** 2
            return loss
        compute_loss_jit = jax.jit(compute_loss)
        # update_fn = lambda pa, si, st: self.optimizer.update(jax.grad(compute_loss)(pa, si), st)

        def update_fn(params_arg, si, opt_stat_arg: optax.GradientTransformation, lr: float):
            opt_stat_arg.hyperparams['learning_rate'] = lr
            g = jax.grad(compute_loss)(params_arg, si)
            updates, opt_stat_arg = self.optimizer.update(g, opt_stat_arg)

            new_params = optax.apply_updates(params_arg, updates)

            # unpacked_state = self.optimizer.unpack_optimizer_state(opt_stat_arg)
            # new_param = unpacked_state.subtree[0]
            params_arg['w'] = softmax_fn(params_arg['w'])
            params_arg['w'] = clip_numeric(params_arg['w'])



            return new_params, opt_stat_arg

        update_fn_jit = jax.jit(update_fn)


        best_sync_data = None
        # for lr in self.learning_rate:
        key, key2 = jax.random.split(key, 2)
        min_loss = 1000000

        for i, param in enumerate(start_params):
            opt_state = self.optimizer.init(param)

            new_param = self.fit_help(param, opt_state, compute_loss_jit, update_fn_jit, clip_numeric, self.learning_rate)
            loss = jnp.linalg.norm(target_stats - diff_stat_fn(softmax_fn(new_param['w'])))
            if loss < min_loss:
                if self.print_progress: print(f'Update results on i={i}.')
                best_sync_data = jnp.copy(new_param['w'])
                min_loss = loss

        self.init_sync_data = jnp.copy(best_sync_data)
        sync_softmax = clip_numeric(softmax_fn(best_sync_data))
        # Dataset.from_onehot_to_dataset(self.domain, best_sync)
        key, key2 = jax.random.split(key, 2)
        oh = Dataset.get_sample_onehot(key2, self.domain, X_relaxed=sync_softmax, num_samples=20)
        return Dataset.from_onehot_to_dataset(self.domain, oh)

    def fit_help(self, params, opt_state, compute_loss_jit, update_fn_jit, post_fn, learning_rates: tuple):

        stop_early = self.stop_early

        self.early_stop_init()
        best_loss = compute_loss_jit(params, 2**15)
        best_loss_last = best_loss
        iters = 0
        t0 = timer()
        t1 = timer()
        # TODO: Save best param
        best_params = None


        if self.print_progress: print(f'Begin training ')
        for lr in learning_rates:
            if self.print_progress: print(f'\tLearning rate={lr}:')

            temp_params = params.copy()
            round_best_loss = compute_loss_jit(temp_params, 2**15)
            for i in range(13):
                # sigmoid = i ** 2
                sigmoid = 2 ** i

                t_sigmoid = timer()
                if self.print_progress: print(f'\t\ti={i:<2}. Sigmoid={sigmoid}: Starting loss={compute_loss_jit(temp_params, 2**15):.5f}')
                loss_hist = []
                t = 0

                for t in range(self.iterations):
                    iters += 1
                    # loss = compute_loss_jit(temp_params, 2**15)
                    loss = compute_loss_jit(temp_params, sigmoid)
                    loss_hist.append(loss)
                    temp_params, opt_state = update_fn_jit(temp_params, sigmoid, opt_state, lr)
                    # temp_params = optax.apply_updates(temp_params, updates)
                    # temp_params['w'] = post_fn(temp_params['w'])
                    # if loss > 2 * best_loss:
                        # if self.print_progress: print(f'\t\t\t# Stop early at {t}. Final loss={loss:.5f}.')
                        # break
                    if (t > stop_early + 1 and len(loss_hist) > 2 * stop_early and loss >= 0.999 * loss_hist[-self.stop_early]):
                        # if self.print_progress: t0 = timer(t0, f'\t\t\t# Stop early at {t}. Final loss={loss:.5f}. time=')
                        break

                    if self.print_progress:
                        total_loss = compute_loss_jit(temp_params, 2**15)
                        if total_loss < 0.95 * round_best_loss:
                            t1 = timer(t1, f'\t\t\tt={t:<5}| total_loss={total_loss:<8.8f}. time=')
                            round_best_loss = total_loss
                # update params:
                this_loss = compute_loss_jit(temp_params, 2**15)
                if self.print_progress:
                    timer(t_sigmoid, f'\t\t\tEnd training at t={t}. Final total_loss={this_loss:<8.8f}, '
                                     f'best_loss={best_loss:8.5f}. time=')
                if this_loss < best_loss - 1e-7:
                    if self.print_progress:
                        print(f'\t\t\t*** This round loss={this_loss:.5f},  best loss={best_loss:.5f}. Updating parameters...')
                    best_loss = this_loss
                    params = temp_params.copy()

        if self.print_progress:
            final_loss = compute_loss_jit(params, 2**15)
            print(f'\tFinal l2 loss = {final_loss}')
            # print(f'\t\t\t*** This round loss={this_loss:.5f},  best loss={best_loss:.5f}. Updating parameters...')

        return params
