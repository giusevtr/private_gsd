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
from models.relaxed_projection import RelaxedProjection
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
            # 0.0001, 0.0002, 0.0003, 0.0004,
            # 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
            # 0.001, 0.002, 0.005,
            0.006,
            # 0.007, 0.01,
            # 0.1,
            # 0.5, 0.7, 0.9,
            # 2, 3
        ]
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
        cat_idx = jnp.concatenate([self.domain.get_attribute_onehot_indices(att).squeeze() for att in self.domain.get_categorical_cols()]).reshape(-1)
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


        target_stats = adaptive_statistic.get_selected_noised_statistics(stat_modules_ids=[1])
        diff_stat_fn = adaptive_statistic.get_selected_statistics_fn(stat_modules_ids=[1])











        # Initialize parameters
        data_dim = self.domain.get_dimension()
        key, key2 = jax.random.split(key, 2)
        init_sync = clip_numeric(softmax_fn(jax.random.uniform(key2, shape=(self.data_size, data_dim), minval=0, maxval=1)))
        init_num, init_cat = separate(init_sync)

        if adaptive_epoch == 1:

            rp = RelaxedProjection(self.domain, self.data_size, self.iterations, self.learning_rate,
                                   self.stop_loss_time_window, self.print_progress)

            data = rp.fit(key, adaptive_statistic)

            self.optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=0)

            cat_stat_fn = adaptive_statistic.get_selected_statistics_fn(stat_modules_ids=[0])
            cat_stats = adaptive_statistic.get_selected_noised_statistics(stat_modules_ids=[0])
            cat_stats_non_priv = adaptive_statistic.get_selected_statistics_without_noise(stat_modules_ids=[0])
            if self.print_progress:
                print(f'Cat Stats Gaussian max_error = ', jnp.abs(cat_stats - cat_stats_non_priv).max())
            def cat_loss_fn(params, numeric_sync, sigmoid):
                cat_sync = params['w']
                sync = join(numeric_sync, cat_sync)
                return jnp.linalg.norm(cat_stat_fn(sync, sigmoid=sigmoid) - cat_stats)

            def update_cat_fn(params_arg, numeric_sync, si, opt_stat_arg: optax.GradientTransformation, lr: float):
                opt_stat_arg.hyperparams['learning_rate'] = lr
                g = jax.grad(cat_loss_fn)(params_arg, numeric_sync, si)
                updates, opt_stat_arg = self.optimizer.update(g, opt_stat_arg)
                new_params = optax.apply_updates(params_arg, updates)
                sync = join(numeric_sync, new_params['w'])
                sync_softmax = softmax_fn(sync)
                new_params['w'] = sync_softmax[:, cat_idx]
                # new_params['w'] = clip_numeric(new_params['w'])
                return new_params, opt_stat_arg

            update_cat_fn_jit = jax.jit(update_cat_fn)

            cat_params = {'w': init_cat}

            if self.print_progress:
                print('train cat params:')
            opt_state = self.optimizer.init(cat_params)

            best_cat_loss = 1000000
            best_cat_param = None
            last_update = 0
            lr = 8
            cat_updates = 10
            for cat_iter in range(5000):
                cat_params, opt_state = update_cat_fn_jit(cat_params, init_num, 2**15, opt_state, lr)
                cat_loss = cat_loss_fn(cat_params, init_num, 2**15)
                if cat_loss < best_cat_loss:
                    if self.print_progress:
                        print(f'{cat_iter:<3}: Cat.Loss = {cat_loss:.8f}')
                    best_cat_loss = cat_loss
                    best_cat_param = cat_params.copy()  # update best parameters.
                    last_update = cat_iter
                if cat_iter > last_update + 20:
                    cat_params = best_cat_param.copy()
                    opt_state = self.optimizer.init(cat_params)
                    lr = lr / 2    # Decrease learning rate
                    cat_updates = cat_updates - 1
                    if self.print_progress:
                        print(f'\tt={cat_iter}. Update learning rate. lr={lr}')
                        print(f'\tCurrent cat.Loss={cat_loss}. Best cat.Loss={best_cat_loss}')
                if cat_updates < 0: # Stop  training
                    break

            sync = join(init_num, best_cat_param['w'])

            if self.print_progress:
                sync_cat_stats = cat_stat_fn(sync)
                error = jnp.abs(cat_stats - sync_cat_stats)
                print('Cat.Errors: ', 'max=', error.max(), '\tmean=', error.mean())

            self.NUM_SYNC_INIT, self.CAT_SYNC_TRAINED = separate(sync)

        # Numeric Loss
        def compute_loss(params, sigmoid):
            num_sync = params['w']
            sync = join(num_sync, self.CAT_SYNC_TRAINED)
            loss = jnp.linalg.norm(diff_stat_fn(sync, sigmoid=sigmoid) - target_stats)
            return loss
        compute_loss_jit = jax.jit(compute_loss)
        def update_fn(params_arg, si, opt_stat_arg: optax.GradientTransformation, lr: float):
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
        key, key2 = jax.random.split(key, 2)
        oh = Dataset.get_sample_onehot(key2, self.domain, X_relaxed=sync_softmax, num_samples=20)
        data_sync = Dataset.from_onehot_to_dataset(self.domain, oh)
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

        for lr_init in learning_rates:
            if self.print_progress:
                print(f'Init.LR ={lr_init}')
                print(f'--------------------------------------------')
            for i in range(5):
                lr = lr_init / 2**i

                for sigmoid in sigmoid_params:
                    temp_params = best_params.copy()
                    opt_state = self.optimizer.init(temp_params)

                    round_best_loss = compute_loss_jit(temp_params, 2**15)

                    init_sig_loss = compute_loss_jit(temp_params, sigmoid)
                    init_loss = compute_loss_jit(temp_params, 2**15)
                    t_sigmoid = timer()
                    loss_hist = [jnp.inf]

                    t = 0
                    best_t = 0
                    parameters_updated = False
                    for t in range(self.iterations):
                        iters += 1

                        temp_params, opt_state = update_fn_jit(temp_params, sigmoid, opt_state, lr)

                        loss = compute_loss_jit(temp_params, 2**15)
                        loss_hist.append(loss)

                        # if (t > stop_early + 1 and len(loss_hist) > 2 * stop_early and loss >=  0.999 * round_best_loss):
                        #     break
                        if (t > stop_early + 1 and loss >= 2.0 * best_loss):
                            break
                        if (t > stop_early + 1 and (t - best_t) > stop_early  and loss >= 0.999 * round_best_loss):
                            break
                        if loss < round_best_loss:
                            best_t = t
                            round_best_loss = loss
                        # Update parameter here:
                        if loss < 0.995 * best_loss - 1e-7:
                            parameters_updated = True
                            best_loss = loss
                            best_params = temp_params.copy()

                    if self.print_progress:
                        this_sig_loss = compute_loss_jit(temp_params, sigmoid)
                        this_loss = compute_loss_jit(temp_params, 2 ** 15)
                        # if True:
                        # if parameters_updated:
                        timer(t_sigmoid,
                              f'\tRound={i:<2}. Sigmoid={sigmoid:<5.0f} and LR={lr:<5.5f}:'
                              f'\t\tStarting Sigmoid.Loss={init_sig_loss:<8.8}'
                                      f'\tLoss={init_loss:<8.8f}'
                              f'\t\t| Final Sigmoid.Loss={this_sig_loss:<8.8f} '
                                         f'\tLoss={this_loss:<8.8f}.'
                                         f'\t*best.Loss={best_loss:<8.8f}*.'
                                         f'\tEnd training at t={t}. '
                                        f'\tparameters_upt={parameters_updated}. '
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
