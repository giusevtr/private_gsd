import chex
import jax
import jax.numpy as jnp
from typing import Callable
from models import Generator
from dataclasses import dataclass
from utils import Dataset, Domain, timer
from stats import Marginals, AdaptiveStatisticState, ChainedStatistics
from jax import jit, value_and_grad
from jax.example_libraries import optimizers


@dataclass
class RelaxedProjectionPPneurips(Generator):
    # domain: Domain
    data_size: int
    iterations: int
    learning_rate: tuple = (0.8,)
    print_progress: bool = False
    early_stop_percent: float = 0.001

    def __init__(self, domain, data_size, iterations=1000, learning_rate=(0.001,),
                 stop_loss_time_window=20, print_progress=False):
        # super().__init__(domain, stat_module, data_size, seed)
        self.domain = domain
        self.data_size = data_size
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.early_stop_percent = 0.001
        self.stop_loss_time_window = stop_loss_time_window
        self.print_progress = print_progress
        self.CACHE = {}
        self.stop_early = 50



    def __str__(self):
        return 'RAP++'

    def fit(self, key, adaptive_statistic: ChainedStatistics, init_data: Dataset=None, tolerance=0, adaptive_epoch=1):

        softmax_fn = lambda X: Dataset.apply_softmax(self.domain, X)
        softmax_fn = jax.jit(softmax_fn)
        data_dim = self.domain.get_dimension()
        key, key2 = jax.random.split(key, 2)

        # Check if this is the first adaptive round. If so, then initialize a synthetic data
        # selected_workloads = len(adaptive_statistic.selected_workloads)
        # if adaptive_epoch == 1:
        #     if self.print_progress: print('Initializing relaxed dataset')
        #     self.init_params = softmax_fn( jax.random.uniform(key2, shape=(self.data_size, data_dim), minval=0, maxval=1))

        D_prime_init = softmax_fn(jax.random.uniform(key2, shape=(self.data_size, data_dim), minval=0, maxval=1))


        target_stats = adaptive_statistic.get_selected_noised_statistics()
        diff_stat_fn = adaptive_statistic.get_selected_statistics_fn()

        numeric_cols = self.domain.get_numeric_cols()
        num_idx = jnp.array([self.domain.get_attribute_onehot_indices(att) for att in numeric_cols]).reshape(-1)

        @jax.jit
        def clip_numeric(D):
            D = D.at[:, num_idx].set(jnp.clip(D[:, num_idx], 0, 1))
            return D

        key, key2 = jax.random.split(key, 2)

        opt_init, opt_update, get_params = optimizers.adam(lambda x: x)
        # D_prime_init = get_params(opt_init)

        @jit
        def loss_fn(D_prime, sigmoid_param):
            stats1 = diff_stat_fn(D_prime, sigmoid=sigmoid_param)
            loss_1 = jnp.linalg.norm(target_stats - stats1, ord=2)
            return loss_1

        @jit
        def update_fn(
                state, sigmoid_param, opt_lr
        ):
            """Compute the gradient and update the parameters"""
            D_prime = get_params(state)
            value, grads = value_and_grad(loss_fn, argnums=0)(
                D_prime, sigmoid_param
            )
            # grads = clip_continuous(grads, feats_idx, -args.clip_grad, args.clip_grad)
            state = opt_update(opt_lr, grads, state)

            unpacked_state = optimizers.unpack_optimizer_state(state)
            new_D_prime = unpacked_state.subtree[0]
            new_D_prime = softmax_fn(new_D_prime)
            new_D_prime = clip_numeric(new_D_prime)
            unpacked_state.subtree = (
                new_D_prime,
                unpacked_state.subtree[1],
                unpacked_state.subtree[2],
            )
            updated_state = optimizers.pack_optimizer_state(unpacked_state)
            return updated_state, get_params(updated_state), value

        D_prime = self.fit_help(D_prime_init, opt_init, loss_fn, update_fn, self.learning_rate[0])
        self.init_params = jnp.copy(D_prime)
        # Dataset.from_onehot_to_dataset(self.domain, best_sync)
        key, key2 = jax.random.split(key, 2)
        oh = Dataset.get_sample_onehot(key2, self.domain, X_relaxed=D_prime, num_samples=20)
        return Dataset.from_onehot_to_dataset(self.domain, oh)

    def fit_help(self, D_prime_init, opt_init, compute_loss_jit, update_fn_jit, opt_lr):

        stop_early = self.stop_early

        self.early_stop_init()
        best_loss = compute_loss_jit(D_prime_init, 2**15)
        iters = 0
        t0 = timer()
        t1 = timer()
        sigmoid_double = 15
        # TODO: Save best param


        D_prime_best = D_prime_init.copy()
        for i in range(9):
            D_prime = D_prime_best.copy()
            opt_state = opt_init(D_prime)

            t_sigmoid = timer()
            # temp_params = params.copy()

            loss_hist = [jnp.inf]
            sigmoid_last_loss = best_loss

            opt_lr_epoch = opt_lr / 2**i

            if self.print_progress: print(f'\tEpoch {i:<3}, lr={opt_lr_epoch:.5f}'
                                          f'\tStarting loss={compute_loss_jit(D_prime, 2**15):.5f}'
                                          f'\tlearning_rate={opt_lr_epoch}')

            sigmoid_param = 1
            sig_counter = 0
            for t in range(self.iterations):
                iters += 1
                # Update
                opt_state, D_prime, loss = update_fn_jit(
                    opt_state, sigmoid_param, opt_lr_epoch
                )

                epoch_loss = float(compute_loss_jit(D_prime, 2**15))
                loss_hist.append(epoch_loss)

                """ Stop early and Save best D_prime """
                D_prime_best, best_loss = (
                    (D_prime.copy(), epoch_loss)
                    if epoch_loss < best_loss
                    else (D_prime_best, best_loss)
                )


                if self.print_progress:
                    # total_loss = compute_loss_jit(D_prime, 2**15)
                    if epoch_loss < 0.95 * sigmoid_last_loss:
                        t1 = timer(t1, f"\t\t# epoch={i:<3}, mini_e={t:<4}:"
                                           f"\tl2-loss-diff={loss:.5f},\tprogress_loss={epoch_loss:.5f},"
                                        f"\tbest_loss={best_loss:.5f}, "
                                           f"\tsig_counter={sig_counter}. time=")
                        sigmoid_last_loss = epoch_loss

                if len(loss_hist) > 10:
                    ave_last = loss_hist[-5]
                    percent_change = (ave_last - epoch_loss) / (ave_last + 1e-9)
                    if percent_change < 0.0001:
                        # if self.print_progress:
                        #     print(
                        #         f"\t\t#Update Sigmoid={sigmoid_param:<5}: "
                        #         f"best_loss={best_loss:.4f}"
                        #     )
                        sigmoid_param = 2 * sigmoid_param
                        sig_counter += 1
                        loss_hist = []
                        if sig_counter > sigmoid_double:
                            break


                        # update params:
            this_loss = compute_loss_jit(D_prime, 2**15)
            if self.print_progress:
                timer(t_sigmoid, f'\t... End training epoch {i:<3}| temp_params.total_loss={this_loss:<8.5f}, best_loss={best_loss:8.5f}. time=')
            # if this_loss < best_loss or D_prime_best is None:
            #     if self.print_progress:
            #         print(f'\t5) Updating parameters...')
            #     best_loss = this_loss
            #     D_prime_best = D_prime.copy()

        return D_prime_best
