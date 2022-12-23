"""
To run parallelization on multiple cores set
XLA_FLAGS=--xla_force_host_platform_device_count=4
"""
import jax
import jax.numpy as jnp
from models_v2 import Generator
import time
from utils import Dataset, Domain
from stats_v2 import Statistic
from models_v2.priv_ga import SimpleGAforSyncData
from dataclasses import dataclass


@dataclass
class PrivGAIP(Generator):
    # domain: Domain
    # stat_module: Statistic
    data_size: int
    # seed: int
    num_generations: int
    popsize: int
    top_k: int
    # crossover: int
    # mutations: int
    stop_loss_time_window: int
    print_progress: bool
    start_mutations: int = None
    regularization_statistics: Statistic = None
    # call_back_fn = None

    def __str__(self):
        reg = self.regularization_statistics is not None
        # return f'SimpleGA(popsize={self.popsize}, topk={self.top_k}, reg={reg})'
        return f'PrivGA(IP)'

    def fit(self, key, stat_module: Statistic, init_X=None):
        """
        Minimize error between real_stats and sync_stats
        """
        # num_queries = stat_module.get_num_queries()
        # indices = jax.random.choice(key, jnp.arange(num_queries), shape=(100, ), )
        # sub_stat = stat_module.get_sub_stat_module(indices)



        all_stat_fn = jax.jit(stat_module.get_stats_fn())
        selected_stat_fn = jax.jit(stat_module.get_sub_stats_fn())
        priv_stats = stat_module.private_stats
        confidence_bound = stat_module.confidence_bound

        # @jax.jit


        key, subkey = jax.random.split(key, 2)
        starting_X = Dataset.synthetic_jax_rng(stat_module.domain, 1000, subkey)
        old_stats = all_stat_fn(starting_X)


        key, subkey = jax.random.split(key, 2)

        # key = jax.random.PRNGKey(seed)
        self.data_dim = stat_module.domain.get_dimension()
        init_time = time.time()
        num_devices = jax.device_count()
        if num_devices > 1:
            print(f'************ {num_devices}  devices found. Using parallelization. ************')

        self.elite_ratio = self.top_k / self.popsize
        strategy = SimpleGAforSyncData(
                    domain=stat_module.domain,
                    data_size=self.data_size,
                    generations=self.num_generations,
                    popsize=self.popsize,
                    elite_ratio=self.elite_ratio,
                    num_devices=num_devices)


        # FITNESS
        # compute_error_fn = lambda X: (jnp.linalg.norm(true_stats - stat_fn(X), ord=2)**2 ).squeeze()

        def count_constrain_fn(X):
            eval = jnp.abs(priv_stats - selected_stat_fn(X))
            cont = jnp.where(eval > confidence_bound, 1, 0)
            return cont
        count_constrain_fn_vmap = jax.vmap(count_constrain_fn, in_axes=(0, ))

        def constrain_fn(X):
            eval = jnp.abs(priv_stats - selected_stat_fn(X))
            cont = jnp.where(eval > confidence_bound, eval , 0)
            return jnp.linalg.norm(self.data_size * cont, ord=2)**2

        constrain_fn_vmap = jax.vmap(constrain_fn, in_axes=(0, ))

        # priv_stats_indices_jnp = jnp.array(stat_module.privately_selected_statistics)
        def compute_error_fn(X):
            old_errors = old_stats - all_stat_fn(X)
            # old_errors = old_errors.at[priv_stats_indices_jnp].set(0)
            # num_old_stats = old_errors.shape[0] - priv_stats_indices_jnp.shape[0]
            temp1 = jnp.linalg.norm(old_errors, ord=2)**2 / old_errors.shape[0]
            # temp1 = jnp.abs(old_errors).max()
            # temp2 = jnp.linalg.norm(self.data_size * constrain_fn(X), ord=1)
            return temp1
        compute_error_vmap = jax.vmap(compute_error_fn, in_axes=(0, ))



        def fitness_fn(x):
            """
            Evaluate the error of the synthetic data
            """
            if num_devices == 1:
                return compute_error_vmap(x)

        stime = time.time()
        self.key, subkey = jax.random.split(key, 2)
        state = strategy.initialize(subkey)
        if self.start_mutations is not None:
            state = state.replace(mutations=self.start_mutations)

        if init_X is not None:
            temp = init_X.reshape((1, init_X.shape[0], init_X.shape[1]))
            new_archive = jnp.concatenate([temp, state.archive[1:, :, :]])
            state = state.replace(archive=new_archive)


        last_fitness = None
        best_fitness_avg = 100000
        last_best_fitness_avg = None

        for t in range(self.num_generations):
            self.key, ask_subkey, eval_subkey = jax.random.split(self.key, 3)
            x, state = strategy.ask(ask_subkey, state)

            # FITNESS
            # fitness = fitness_fn(x)
            # reg_score = compute_error_vmap(x)
            const_score = constrain_fn_vmap(x)
            fitness =  const_score

            state = strategy.tell(x, fitness, state)

            best_fitness = fitness.min()

            # Early stop
            best_fitness_avg = min(best_fitness_avg, best_fitness)

            num_constrains = jnp.sum(count_constrain_fn(state.best_member)).astype(int)

            if num_constrains == 0 :
                print(f'\t\tEarly stop at {t}. time = {time.time() - stime:.2f}.')
                break
            # if num_constrains == 0 and t % self.stop_loss_time_window == 0 and t > 0:
                # if last_best_fitness_avg is not None:
                #     percent_change = jnp.abs(best_fitness_avg - last_best_fitness_avg) / last_best_fitness_avg
                #     if percent_change < 0.01:
                #         print(f'\t\tEarly stop at {t}. time = {time.time() - stime:.2f}.')
                #         break
                # last_best_fitness_avg = best_fitness_avg
                # best_fitness_avg = 100000

            if last_fitness is None or best_fitness < last_fitness * 0.95 or t > self.num_generations-2 :
            # if  t % 5 == 0:
                if self.print_progress:
                    X = state.best_member
                    error = jnp.abs(priv_stats - selected_stat_fn(X))
                    print(f'\tGeneration {t:04}, best_l2_fitness = {jnp.sqrt(best_fitness):.5f}, ', end=' ')
                    print(f'\tAverage error {jnp.linalg.norm(error, ord=1)/error.shape[0]: .6f}', end=' ')
                    print(f'\tMax error     {error.max():.6f}', end=' ')
                    print(f'\ttime={time.time() -init_time:.3f}(s):', end='')
                    print(f'\tmutations={state.mutations}', end='')

                    # print(f'\nt={t:03}:', end='. ')
                    print(f'\tConstrain counts ', num_constrains,
                        '. const_score = ', constrain_fn(X),
                        '. reg_score = ', compute_error_fn(X),
                          end=' '
                        )

                    print()


                last_fitness = best_fitness

        X_sync = state.best_member
        sync_dataset = Dataset.from_numpy_to_dataset(stat_module.domain, X_sync)
        return sync_dataset


######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################

