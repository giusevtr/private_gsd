"""
To run parallelization on multiple cores set
XLA_FLAGS=--xla_force_host_platform_device_count=4
"""
import jax
import jax.numpy as jnp
from models import Generator
import time
from utils import Dataset, Domain
from stats import Statistic
from visualize.plot_low_dim_data import plot_2d_data


class PrivGA(Generator):
    # domain: Domain
    # stat_module: Statistic
    # data_size: int
    # seed: int
    # num_generations: int
    # popsize: int
    # top_k: int
    # crossover: int
    # mutations: int,
    # stop_loss_time_window: int,
    # splits: int,
    def __init__(self,
                 domain: Domain,
                 stat_module: Statistic,
                 data_size,
                 seed,
                 num_generations: int,
                 popsize: int,
                 top_k: int,
                 stop_loss_time_window:int,
                 print_progress: bool,
                 start_mutations:int = None,
                 ):

        # Instantiate the search strategy
        super().__init__(domain, stat_module, data_size, seed)
        self.num_generations = num_generations
        self.popsize = popsize
        self.top_k = top_k
        self.elite_ratio = top_k / popsize
        self.print_progress = print_progress
        self.stop_loss_time_window = stop_loss_time_window
        self.start_mutations = start_mutations
        num_devices = jax.device_count()

        self.strategy = SimpleGAforSyncData(
            domain=domain,
            data_size=data_size,
            generations=num_generations,
            popsize=self.popsize,
            elite_ratio=self.elite_ratio,
            num_devices=num_devices)

        # Initialize statistics
        # self.stat_fn = jax.jit(stat_module.get_stats_fn())
        self.stat_fn = stat_module.get_stats_fn()

        # self.init_population_fn = jax.jit(get_initialize_population)

    @staticmethod
    def get_generator(num_generations=100, popsize=20, top_k=5,
                       clip_max=1, stop_loss_time_window=10,  print_progress=False, start_mutations: int = None):
        generator_init = lambda domain, stat_module, data_size, seed: PrivGA(domain,
                                                                            stat_module,
                                                                            data_size,
                                                                            seed,
                                                                            num_generations=num_generations,
                                                                            popsize=popsize,
                                                                            top_k=top_k,
                                                                            stop_loss_time_window=stop_loss_time_window,
                                                                            print_progress=print_progress,
                                                                             start_mutations=start_mutations)
        return generator_init

    def __str__(self):
        return f'SimpleGA(popsize={self.popsize}, topk={self.top_k})'

    def fit(self, true_stats, init_X=None):
        """
        Minimize error between real_stats and sync_stats
        """
        init_time = time.time()
        num_devices = jax.device_count()
        if num_devices > 1:
            print(f'************ {num_devices}  devices found. Using parallelization. ************')

        # FITNESS
        compute_error_fn = lambda X: (jnp.linalg.norm(true_stats - self.stat_fn(X), ord=2)**2 ).squeeze()
        # compute_error_vmap = jax.jit(jax.vmap(compute_error_fn, in_axes=(0, )))
        compute_error_vmap = jax.vmap(compute_error_fn, in_axes=(0, ))

        def distributed_error_fn(X):
            return compute_error_vmap(X)
        compute_error_pmap = jax.pmap(distributed_error_fn, in_axes=(0, ))

        def fitness_fn(x):
            """
            Evaluate the error of the synthetic data
            """
            if num_devices == 1:
                return compute_error_vmap(x)

            X_distributed = x.reshape((num_devices, -1, self.data_size, self.data_dim))
            fitness = compute_error_pmap(X_distributed)
            fitness = jnp.concatenate(fitness)
            return fitness.squeeze()

        stime = time.time()
        self.key, subkey = jax.random.split(self.key, 2)
        state = self.strategy.initialize(subkey)
        if self.start_mutations is not None:
            state = state.replace(mutations=self.start_mutations)

        # if init_X is not None:
        #     state.archive = jnp.stack((init_X, state.archive[1:, :, :]))
        last_fitness = None
        best_fitness_avg = 100000
        last_best_fitness_avg = None

        # MUT_UPT_CNT = 10000 // self.popsize
        MUT_UPT_CNT = 3
        counter = 0
        for t in range(self.num_generations):
            self.key, ask_subkey, eval_subkey = jax.random.split(self.key, 3)
            x, state = self.strategy.ask(ask_subkey, state)

            # FITNESS
            fitness = fitness_fn(x)

            state = self.strategy.tell(x, fitness, state)
            best_fitness = fitness.min()

            if best_fitness > best_fitness_avg:
                counter = counter + 1
                if counter >= MUT_UPT_CNT:
                    state = state.replace(mutations=(state.mutations + 1) // 2)
                    counter = 0

            # Early stop
            best_fitness_avg = min(best_fitness_avg, best_fitness)

            plot_2d_data(state.best_member, title=f'g={t}, fitness={best_fitness:.2f}', save_path=f'progress/{t:04}.png')

            if t % self.stop_loss_time_window == 0 and t > 0:
                if last_best_fitness_avg is not None:
                    percent_change = jnp.abs(best_fitness_avg - last_best_fitness_avg) / last_best_fitness_avg
                    if percent_change < 0.001:
                        break

                last_best_fitness_avg = best_fitness_avg
                best_fitness_avg = 100000

            if last_fitness is None or best_fitness < last_fitness * 0.95 or t > self.num_generations-2 :
                if self.print_progress:
                    X_sync = state.best_member
                    errors = true_stats - self.stat_fn(X_sync)
                    max_error = jnp.abs(errors).max()

                    print(f'\tGeneration {t}, best_l2_fitness = {jnp.sqrt(best_fitness):.3f}, ', end=' ')
                    print(f'\ttime={time.time() -init_time:.3f}(s):', end='')
                    print(f'\t\tmax_error={max_error:.3f}', end='')
                    print(f'\tmutations={state.mutations}', end='')
                    print()


                last_fitness = best_fitness

        X_sync = state.best_member
        sync_dataset = Dataset.from_numpy_to_dataset(self.domain, X_sync)
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








import jax
import numpy as np
import chex
from flax import struct
from utils import Dataset, Domain
from functools import partial
from typing import Tuple, Optional
from evosax.utils import get_best_fitness_member

@struct.dataclass
class EvoState:
    mean: chex.Array
    archive: chex.Array
    fitness: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0
    mutations: int = 1

"""
Implement crossover that is specific to synthetic data
"""
class SimpleGAforSyncData:
    def __init__(self,
                 domain: Domain,
                 data_size: int,  # number of synthetic data rows
                 generations: int,
                 popsize: int,
                 elite_ratio: float = 0.5,
                 num_devices=1,
                 ):
        """Simple Genetic Algorithm For Synthetic Data Search Adapted from (Such et al., 2017)
        Reference: https://arxiv.org/abs/1712.06567
        Inspired by: https://github.com/hardmaru/estool/blob/master/es.py"""

        d = len(domain.attrs)
        num_dims = d * data_size
        self.data_size = data_size
        self.popsize = popsize
        # super().__init__(num_dims, popsize)
        self.domain = domain
        self.generations = generations
        # self.sync_data_shape = sync_data_shape
        self.elite_ratio = elite_ratio
        self.elite_popsize = max(1, int(self.popsize * self.elite_ratio))
        self.strategy_name = "SimpleGA"
        # self.crossover = crossover
        # self.mutations = mutations
        self.num_devices = num_devices
        self.domain = domain

        mutate = get_mutation_fn(domain)
        self.mutate_vmap = jax.jit(jax.vmap(mutate, in_axes=(0, 0, None)))

        self.mate_vmap = jax.jit(jax.vmap(single_mate, in_axes=(0, 0, 0)))

    # @partial(jax.jit, static_argnums=(0,))
    def initialize(
        self, rng: chex.PRNGKey
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        # Initialize strategy based on strategy-specific initialize method
        state = self.initialize_strategy(rng)
        return state

    def initialize_strategy(self, rng: chex.PRNGKey) -> EvoState:
        """`initialize` the differential evolution strategy."""
        initialization = initialize_population(rng, self.elite_popsize, self.domain, self.data_size).astype(jnp.float32)

        state = EvoState(
            mean=initialization.mean(axis=0),
            archive=initialization,
            fitness=jnp.zeros(self.elite_popsize) + jnp.finfo(jnp.float32).max,
            best_member=initialization[0],
            mutations=self.data_size
        )
        return state

    @partial(jax.jit, static_argnums=(0,))
    def ask(
        self,
        rng: chex.PRNGKey,
        state: EvoState,
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        x, state = self.ask_strategy(rng, state)
        return x, state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState
    ) -> Tuple[chex.Array, EvoState]:
        rng, rng_a, rng_b, rng_mate, rng_2 = jax.random.split(rng, 5)
        elite_ids = jnp.arange(self.elite_popsize)



        idx_a = jax.random.choice(rng_a, elite_ids, (self.popsize // 2,))
        idx_b = jax.random.choice(rng_b, elite_ids, (self.popsize // 2,))
        A = state.archive[idx_a]
        B = state.archive[idx_b]

        rng_mate_split = jax.random.split(rng_mate, self.popsize // 2)
        C = self.mate_vmap(rng_mate_split, A, B)

        x = jnp.concatenate((A, C))

        rng_mutate = jax.random.split(rng_2, self.popsize)
        x = self.mutate_vmap(rng_mutate, x, state.mutations)
        return x.astype(jnp.float32), state


    @partial(jax.jit, static_argnums=(0,))
    def tell(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
    ) -> EvoState:
        """`tell` performance data for strategy state update."""

        # Update the search state based on strategy-specific update
        state = self.tell_strategy(x, fitness, state)

        # Check if there is a new best member & update trackers
        best_member, best_fitness = get_best_fitness_member(x, fitness, state)
        return state.replace(
            best_member=best_member,
            best_fitness=best_fitness,
            gen_counter=state.gen_counter + 1,
        )

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
    ) -> EvoState:
        """
        `tell` update to ES state.
        If fitness of y <= fitness of x -> replace in population.
        """
        # Combine current elite and recent generation info
        fitness = jnp.concatenate([fitness, state.fitness])
        solution = jnp.concatenate([x, state.archive])
        # Select top elite from total archive info
        idx = jnp.argsort(fitness)[0: self.elite_popsize]

        ## MODIFICATION: Select random survivors
        fitness = fitness[idx]
        archive = solution[idx]
        # Update mutation epsilon - multiplicative decay

        # Keep mean across stored archive around for evaluation protocol
        mean = archive.mean(axis=0)
        return state.replace(
            fitness=fitness, archive=archive,  mean=mean
        )


def initialize_population(rng: chex.PRNGKey, pop_size, domain: Domain, data_size):
    temp = []
    for s in range(pop_size):
        rng, rng_sub = jax.random.split(rng)
        X = Dataset.synthetic_jax_rng(domain, data_size, rng_sub)
        temp.append(X)

    initialization = jnp.array(temp)
    return initialization


def get_mutation_fn(domain: Domain):

    def mutate(rng: chex.PRNGKey, X, mutations):
        n, d = X.shape
        rng1, rng2 = jax.random.split(rng, 2)
        total_params = n * d

        # mut_coordinates = jnp.array([i < mutations for i in range(total_params)])
        mut_coordinates = jnp.arange(total_params) < mutations
        # idx = jnp.concatenate((jnp.ones(mutations), jnp.zeros(total_params-mutations)))
        mut_coordinates = jax.random.permutation(rng1, mut_coordinates)
        mut_coordinates = mut_coordinates.reshape((n, d))
        initialization = Dataset.synthetic_jax_rng(domain, n, rng2)
        X = X * (1-mut_coordinates) + initialization * mut_coordinates
        return X

    return mutate

######################################################################
######################################################################
######################################################################
######################################################################
######################################################################


def test_mutation():

    domain = Domain(['A', 'B', 'C'], [10, 10, 1])

    mutate = jax.jit(get_mutation_fn(domain))
    rng = jax.random.PRNGKey(2)
    # x = initialize_population(rng, pop_size=3, domain=domain, data_size=4)

    x = Dataset.synthetic_jax_rng(domain, 4, rng)

    rng, rng_sub = jax.random.split(rng)
    x2 = mutate(rng_sub, x, 2)

    print('x =')
    print(x)
    print('x2=')
    print(x2)

def single_mate(
    rng: chex.PRNGKey, a: chex.Array, b: chex.Array,
) -> chex.Array:
    """Only cross-over dims for x% of all dims."""
    n, d = a.shape
    # n, d = sync_data_shape

    X = a
    Y = b

    # rng1, rng2 = jax.random.split(rng, 2)
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
    cross_over_rate = 0.1 * jax.random.uniform(rng1, shape=(1,))

    idx = (jax.random.uniform(rng2, (n, )) > cross_over_rate).reshape((n, 1))
    X = jax.random.permutation(rng3, X, axis=0)
    Y = jax.random.permutation(rng4, Y, axis=0)

    XY = X * (1 - idx) + Y * idx
    cross_over_candidate = XY
    return cross_over_candidate

if __name__ == "__main__":
    # test_crossover()

    test_mutation()