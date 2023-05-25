import itertools

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

        self.learning_rate = [
            0.00000001,
            0.00000005,
            0.0000001,
            0.0000005,
            0.000001,
            0.00001, 0.00005,
            0.0001, 0.0002, 0.0003, 0.0004,
                   0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.005, 0.007, 0.01, 0.1, 0.5, 0.7, 0.9]

        # self.learning_rate = [0.1]

        self.learning_rate.reverse()

    def __str__(self):
        return 'RAP++_old'

    # def init_categorical_stats(self,
    #     categorical_statistics: ChainedStatistics):
    #     self.categorical = categorical_statistics
    #     self.categorical.mea

    def fit(self, key,
            adaptive_statistic: ChainedStatistics, init_data: Dataset=None, tolerance=0, adaptive_epoch=1):

        softmax_fn = lambda X: Dataset.apply_softmax(self.domain, X)
        numeric_cols = self.domain.get_numeric_cols()
        num_idx = jnp.array([self.domain.get_attribute_onehot_indices(att) for att in numeric_cols]).reshape(-1)
        cat_idx = jnp.array([self.domain.get_attribute_onehot_indices(att) for att in self.domain.get_categorical_cols()]).reshape(-1)

        stat_fn = adaptive_statistic.get_selected_statistics_fn()
        @jax.jit
        def clip_numeric(D):
            D = D.at[:, num_idx].set(jnp.clip(D[:, num_idx], 0, 1))
            return D

        softmax_fn = jax.jit(softmax_fn)
        data_dim = self.domain.get_dimension()
        key, key2 = jax.random.split(key, 2)

        # Check if this is the first adaptive round. If so, then initialize a synthetic data
        # selected_workloads = len(adaptive_statistic.selected_workloads)
        start_params = []
        init_params_1 = {'w': clip_numeric(softmax_fn(jax.random.uniform(key2, shape=(self.data_size, data_dim), minval=0, maxval=1)))}
        start_params.append(init_params_1)
        # if adaptive_epoch > 1:
            # if self.print_progress: print('Initializing relaxed dataset')
            # start_params.append({'w': self.init_sync_data})
            # self.init_params = jax.random.uniform(key2, shape=(self.data_size, data_dim), minval=0, maxval=1)
        # params = {'w': self.init_params.copy()}


        target_stats = adaptive_statistic.get_selected_noised_statistics()
        diff_stat_fn = adaptive_statistic.get_selected_statistics_fn()



        self.optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=0)

        workload_fn_list = []
        stat_mod = adaptive_statistic.stat_modules[0]
        workload_ids = adaptive_statistic.get_selected_workload_ids(0)
        if workload_ids.shape[0] > 0:
            workload_fn_list.append(stat_mod._get_workload_fn(workload_ids))
        def cat_stat_fn(X, **kwargs):
            return jnp.concatenate([fn(X, **kwargs) for fn in workload_fn_list], axis=0)

        selected_chained_stats = []
        temp = [selected[2] for selected in adaptive_statistic.selected_workloads[0]]
        if len(temp) > 0:
            temp = jnp.concatenate(temp)
            selected_chained_stats.append(temp)
        cat_stats = jnp.concatenate(selected_chained_stats)

        def cat_loss_fn(params, sigmoid):
            return jnp.linalg.norm(cat_stat_fn(params['w'], sigmoid=sigmoid) - cat_stats)

        def update_cat_fn(params_arg, si, opt_stat_arg: optax.GradientTransformation, lr: float):
            opt_stat_arg.hyperparams['learning_rate'] = lr
            g = jax.grad(cat_loss_fn)(params_arg, si)
            # Zero out numerical gradients
            g['w'] = g['w'].at[:, num_idx].set(0)
            updates, opt_stat_arg = self.optimizer.update(g, opt_stat_arg)
            new_params = optax.apply_updates(params_arg, updates)
            new_params['w'] = softmax_fn(new_params['w'])
            # new_params['w'] = clip_numeric(new_params['w'])
            return new_params, opt_stat_arg

        def compute_loss(params, sigmoid):
            w = params['w']
            loss = jnp.linalg.norm(diff_stat_fn(w, sigmoid=sigmoid) - target_stats)
            return loss
        compute_loss_jit = jax.jit(compute_loss)

        def update_fn(params_arg, si, opt_stat_arg: optax.GradientTransformation, lr: float):
            print('compiling update_fn')
            opt_stat_arg.hyperparams['learning_rate'] = lr
            g = jax.grad(compute_loss)(params_arg, si)
             # Zero out real-value features gradients
            g['w'] = g['w'].at[:, cat_idx].set(0)

            updates, opt_stat_arg = self.optimizer.update(g, opt_stat_arg)
            new_params = optax.apply_updates(params_arg, updates)
            # new_params['w'] = softmax_fn(new_params['w'])
            new_params['w'] = clip_numeric(new_params['w'])
            return new_params, opt_stat_arg


        update_fn_jit = jax.jit(update_fn)
        update_cat_fn_jit = jax.jit(update_cat_fn)


        best_sync_data = None
        key, key2 = jax.random.split(key, 2)
        min_loss = 1000000

        for i, param in enumerate(start_params):
            new_param = self.fit_help(param, compute_loss_jit, update_fn_jit, cat_loss_fn, update_cat_fn_jit, self.learning_rate, cat_stat_fn)
            loss = jnp.linalg.norm(target_stats - diff_stat_fn(new_param['w']))
            if loss < min_loss:
                if self.print_progress: print(f'Update results on i={i}.')
                best_sync_data = jnp.copy(new_param['w'])
                min_loss = loss

        self.init_sync_data = jnp.copy(best_sync_data)
        sync_softmax = jnp.copy(best_sync_data)
        # Dataset.from_onehot_to_dataset(self.domain, best_sync)
        # key, key2 = jax.random.split(key, 2)
        # oh = Dataset.get_sample_onehot(key2, self.domain, X_relaxed=sync_softmax, num_samples=20)
        data_sync = Dataset.from_onehot_to_dataset(self.domain, sync_softmax)
        # data_np_oh = data_sync.to_onehot()
        # loss_post = compute_loss_jit(data_np_oh, 2**15)
        # print(f'Debug: final output loss = {loss_post}')
        return data_sync

    def fit_help(self, params, compute_loss_jit, update_fn_jit, cat_loss_fn, update_cat_fn_jit, learning_rates: list, stat_fn):

        stop_early = self.stop_early

        self.early_stop_init()
        iters = 0
        t1 = timer()

        if self.print_progress: print(f'Begin training ')

        sigmoid_params = [2** i for i in range(16)]

        print('train cat params:')
        temp_params = params.copy()
        opt_state = self.optimizer.init(temp_params)

        best_cat_loss = 1000000
        for cat_iter in range(500):
            temp_params, opt_state = update_cat_fn_jit(temp_params, 2**15, opt_state, 3.0)
            cat_loss = cat_loss_fn(temp_params, 2**15)
            if cat_loss < best_cat_loss :
                total_loss = compute_loss_jit(temp_params, 2**15)
                print(f'{cat_iter:<3}: Cat.Loss = {cat_loss:.8f}, Total.Loss={total_loss}')
                best_cat_loss = cat_loss
                params = temp_params.copy()


        best_loss = compute_loss_jit(params, 2**15)
        print('Cat.Stats = ', stat_fn(params['w']), 'Best.Loss', best_loss)

        for i in range(3):

            for sigmoid, lr in itertools.product(sigmoid_params, learning_rates):
                temp_params = params.copy()
                opt_state = self.optimizer.init(temp_params)
                round_best_loss = compute_loss_jit(temp_params, 2**15)

                init_sig_loss = compute_loss_jit(temp_params, sigmoid)
                init_loss = compute_loss_jit(temp_params, 2**15)
                t_sigmoid = timer()
                loss_hist = [jnp.inf]

                t = 0

                parameters_updated = False
                for t in range(self.iterations):
                    iters += 1

                    temp_params, opt_state = update_fn_jit(temp_params, sigmoid, opt_state, lr)
                    # loss1 = compute_loss_jit(temp_params, 2**15)


                    loss = compute_loss_jit(temp_params, 2**15)
                    loss_hist.append(loss)

                    if (t > stop_early + 1 and len(loss_hist) > 2 * stop_early and loss >=  0.9999 * round_best_loss):
                        break
                    if (t > stop_early + 1 and loss >= 2.0 * best_loss):
                        break

                    # Update parameter here:
                    cat_loss = compute_loss_jit(temp_params, 2**15)
                    if cat_loss < best_loss - 1e-7:
                        parameters_updated = True
                        best_loss = cat_loss
                        params = temp_params.copy()

                if self.print_progress:
                    this_sig_loss = compute_loss_jit(temp_params, sigmoid)
                    this_loss = compute_loss_jit(temp_params, 2 ** 15)

                    if parameters_updated:

                        # print(f'\ti={i:<2}. Sigmoid={sigmoid} and lr={lr}.')
                        # print(f'\ti={i:<2}. Sigmoid={sigmoid} and lr={lr}.'
                        #       f'\tStarting Sigmoid-loss={init_sig_loss:8.8}'
                        #               f'\tLoss={init_loss:8.8f}')
                        timer(t_sigmoid,
                                f'\ti={i:<2}. Sigmoid={sigmoid} and lr={lr}.'
                                f'\tStarting Sigmoid-loss={init_sig_loss:8.8}'
                                f'\tLoss={init_loss:8.8f}'
                                f'\t   Final Sigmoid-loss={this_sig_loss:8.8f} '
                                         f'\tLoss={this_loss:8.8f}.'
                                         f'\tbest_loss={best_loss:8.8f}.'
                                         f'\tEnd training at t={t}. '
                                         f' time=')
                    if parameters_updated:
                        print(f'\t\t\t*** Parameters updated ***')



        print('Cat.Stats = ', stat_fn(params['w']))
        if self.print_progress:
            final_loss = compute_loss_jit(params, 2**15)
            print(f'\tFinal l2 loss = {final_loss:.8f}')
            # print(f'\t\t\t*** This round loss={this_loss:.5f},  best loss={best_loss:.5f}. Updating parameters...')

        return params
