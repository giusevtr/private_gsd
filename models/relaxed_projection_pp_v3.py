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
class RelaxedProjectionPP_v3(Generator):
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
            # 0.00000001,
            # 0.00000005,
            # 0.0000001,
            # 0.0000005,
            # 0.000001,
            # 0.00001, 0.00005,
            0.0001, 0.0002, 0.0003, 0.0004,
            0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.005, 0.007, 0.01, 0.1, 0.5
            # , 0.7, 0.9, 2, 3
        ]
        self.learning_rate.reverse()

    def __str__(self):
        return 'RAP++'

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
        num_size = num_idx.shape[0]
        cat_size = cat_idx.shape[0]


        @jax.jit
        def clip_numeric(D):
            D = D.at[:, num_idx].set(jnp.clip(D[:, num_idx], 0, 1))
            return D

        @jax.jit
        def clip(D):
            D = D.at[:, num_idx].set(jnp.clip(D[:, num_idx], 0, 1))
            return D

        def separate(sync):
            num_sync = sync[:, num_idx]
            cat_sync = sync[:, cat_idx]
            return num_sync, cat_sync
        def join(num_sync, cat_sync):
            temp = jnp.column_stack((num_sync, cat_sync))
            temp = temp.at[:, num_idx].set(num_sync)
            temp = temp.at[:, cat_idx].set(cat_sync)
            return temp



        softmax_fn = jax.jit(softmax_fn)
        # if adaptive_epoch > 1:
            # if self.print_progress: print('Initializing relaxed dataset')
            # start_params.append({'w': self.init_sync_data})
            # self.init_params = jax.random.uniform(key2, shape=(self.data_size, data_dim), minval=0, maxval=1)
        # params = {'w': self.init_params.copy()}


        target_stats = adaptive_statistic.get_selected_noised_statistics()
        diff_stat_fn = adaptive_statistic.get_selected_statistics_fn()


        # Initialize parameters
        data_dim = self.domain.get_dimension()
        key, key2 = jax.random.split(key, 2)
        init_sync = clip_numeric(softmax_fn(jax.random.uniform(key2, shape=(self.data_size, data_dim), minval=0, maxval=1)))
        init_num, init_cat = separate(init_sync)

        if adaptive_epoch == 1:
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

            def cat_loss_fn(params, numeric_sync, sigmoid):
                cat_sync = params['w']
                sync = join(numeric_sync, cat_sync)
                return jnp.linalg.norm(cat_stat_fn(sync, sigmoid=sigmoid) - cat_stats)

            def update_cat_fn(params_arg, numeric_sync, si, opt_stat_arg: optax.GradientTransformation, lr: float):
                opt_stat_arg.hyperparams['learning_rate'] = lr
                g = jax.grad(cat_loss_fn)(params_arg, numeric_sync, si)
                # Zero out numerical gradients
                # g['w'] = g['w'].at[:, num_idx].set(0)
                updates, opt_stat_arg = self.optimizer.update(g, opt_stat_arg)
                new_params = optax.apply_updates(params_arg, updates)

                sync = join(numeric_sync, new_params['w'])
                sync_softmax = softmax_fn(sync)

                new_params['w'] = sync_softmax[:, cat_idx]
                # new_params['w'] = clip_numeric(new_params['w'])
                return new_params, opt_stat_arg

            update_cat_fn_jit = jax.jit(update_cat_fn)

            cat_params = {'w': init_cat}

            print('train cat params:')
            opt_state = self.optimizer.init(cat_params)

            best_cat_loss = 1000000
            best_cat_param = None
            for cat_iter in range(500):
                cat_params, opt_state = update_cat_fn_jit(cat_params, init_num, 2**15, opt_state, 3.0)
                cat_loss = cat_loss_fn(cat_params, init_num, 2**15)
                if cat_loss < best_cat_loss:
                    print(f'{cat_iter:<3}: Cat.Loss = {cat_loss:.8f}')
                    best_cat_loss = cat_loss
                    best_cat_param = cat_params.copy()


            # best_loss = compute_loss_jit(best_cat_param, init_num,  2**15)
            sync = join(init_num, best_cat_param['w'])
            print('Cat.Stats = ', cat_stat_fn(sync))

            oh = Dataset.get_sample_onehot(key2, self.domain, X_relaxed=sync, num_samples=20)

            self.NUM_SYNC_INIT, self.CAT_SYNC_TRAINED = init_num, best_cat_param['w']
            # self.NUM_SYNC_INIT, self.CAT_SYNC_TRAINED = separate(oh)

        # Numeric Loss
        def compute_loss(params, sigmoid):
            num_sync = params['w']
            sync = join(num_sync, self.CAT_SYNC_TRAINED)
            loss = jnp.linalg.norm(diff_stat_fn(sync, sigmoid=sigmoid) - target_stats)
            return loss
        compute_loss_jit = jax.jit(compute_loss)
        def update_fn(params_arg, si, opt_stat_arg: optax.GradientTransformation, lr: float):
            print('compiling update_fn')
            opt_stat_arg.hyperparams['learning_rate'] = lr
            g = jax.grad(compute_loss)(params_arg, si)
            updates, opt_stat_arg = self.optimizer.update(g, opt_stat_arg)
            new_params = optax.apply_updates(params_arg, updates)
            new_params['w'] = jnp.clip(new_params['w'], 0, 1)
            return new_params, opt_stat_arg
        update_fn_jit = jax.jit(update_fn)



        start_params = []
        num_params = {'w': self.NUM_SYNC_INIT}
        key, key2 = jax.random.split(key, 2)
        new_param = self.fit_help(num_params, compute_loss_jit, update_fn_jit, self.learning_rate)

        best_sync_data = join(new_param['w'], self.CAT_SYNC_TRAINED)

        self.init_sync_data = jnp.copy(best_sync_data)
        sync_softmax = jnp.copy(best_sync_data)
        # Dataset.from_onehot_to_dataset(self.domain, best_sync)
        key, key2 = jax.random.split(key, 2)
        oh = Dataset.get_sample_onehot(key2, self.domain, X_relaxed=sync_softmax, num_samples=20)
        data_sync = Dataset.from_onehot_to_dataset(self.domain, oh)
        # data_np_oh = data_sync.to_onehot()
        # loss_post = compute_loss_jit(data_np_oh, 2**15)
        # print(f'Debug: final output loss = {loss_post}')
        return data_sync

    def fit_help(self, params, compute_loss_jit, update_fn_jit, learning_rates: list):

        stop_early = self.stop_early

        self.early_stop_init()
        iters = 0
        t1 = timer()
        best_loss = compute_loss_jit(params, 2**15)

        if self.print_progress: print(f'Begin training ')

        sigmoid_params = [2** i for i in range(16)]
        best_params = params.copy()

        for i in range(1):

            for sigmoid in sigmoid_params:

                for lr in learning_rates:

                    temp_params = best_params.copy()
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

                        loss = compute_loss_jit(temp_params, 2**15)
                        loss_hist.append(loss)

                        if (t > stop_early + 1 and len(loss_hist) > 2 * stop_early and loss >=  0.999 * round_best_loss):
                            break
                        if (t > stop_early + 1 and loss >= 2.0 * best_loss):
                            break

                        # Update parameter here:
                        if loss < 0.995 * best_loss - 1e-7:
                            parameters_updated = True
                            best_loss = loss
                            best_params = temp_params.copy()

                    if self.print_progress:
                        this_sig_loss = compute_loss_jit(temp_params, sigmoid)
                        this_loss = compute_loss_jit(temp_params, 2 ** 15)
                        # if True:
                        if parameters_updated:
                            timer(t_sigmoid,
                                  f'\tRound={i:<2}. Sigmoid={sigmoid:<5} and lr={lr:.5}:'
                                  f'\t\tStarting Sigmoid.loss={init_sig_loss:8.8}'
                                          f'\tLoss={init_loss:8.8f}'
                                  f'\t\t| Final Sigmoid.loss={this_sig_loss:8.8f} '
                                             f'\tLoss={this_loss:8.8f}.'
                                             f'\t*best.Loss={best_loss:8.8f}*.'
                                             f'\tEnd training at t={t}. '
                                             f' time=')
                    # if t == self.iterations -1:
                    #     if self.print_progress: print('\t\tBreaking out of the lr loop.')
                    #     break

                    # if self.print_progress:
                    #     final_loss = compute_loss_jit(best_params, 2**15)
                    #     print(f'\tRound {i} and sigmoid={sigmoid}, lr={lr:.5f}: start.Loss={round_start_loss:.8f},  final.Loss = {final_loss:.8f}')

        if self.print_progress:
            final_loss = compute_loss_jit(best_params, 2**15)
            print(f'\tFinal l2 loss = {final_loss:.8f}')
            # print(f'\t\t\t*** This round loss={this_loss:.5f},  best loss={best_loss:.5f}. Updating parameters...')

        return best_params
