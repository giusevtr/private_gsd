"""
To run parallelization on multiple cores set
XLA_FLAGS=--xla_force_host_platform_device_count=4
"""
import jax
import jax.numpy as jnp
from models_v3 import Generator
import time
from utils import Dataset, Domain
from stats_v3 import Marginals, PrivateMarginalsState

from dataclasses import dataclass


# @dataclass
class PrivGA(Generator):
    data_size: int
    num_generations: int
    popsize: int
    top_k: int
    stop_loss_time_window: int
    print_progress: bool
    start_mutations: int = None

    def __init__(self, domain, data_size, num_generations, popsize, top_k, stop_loss_time_window, print_progress, start_mutations):
        self.domain = domain
        elite_ratio = top_k / popsize
        num_devices = jax.device_count()
        self.data_size = data_size
        self.num_generations = num_generations
        self.stop_loss_time_window = stop_loss_time_window
        self.print_progress = print_progress
        self.start_mutations = start_mutations

        self.strategy = SimpleGAforSyncData(
            domain=domain,
            data_size=data_size,
            generations=num_generations,
            popsize=popsize,
            elite_ratio=elite_ratio,
            num_devices=num_devices)

    def __str__(self):
        return f'PrivGA'

    def fit(self, key, stat: PrivateMarginalsState, init_X=None, tolerance=0):
        """
        Minimize error between real_stats and sync_stats
        """
        # stat_fn = jax.jit(stat_module.get_sub_stats_fn())
        key, key_sub = jax.random.split(key, 2)

        # key = jax.random.PRNGKey(seed)
        init_time = time.time()
        num_devices = jax.device_count()
        if num_devices > 1:
            print(f'************ {num_devices}  devices found. Using parallelization. ************')

        # FITNESS
        compute_error_fn = lambda X: stat.priv_loss_l2(X)
        compute_error_vmap = jax.vmap(compute_error_fn, in_axes=(0, ))
        fitness_fn = lambda x: compute_error_vmap(x)
        # fitness_fn = jax.jit(fitness_fn)

        self.key, subkey = jax.random.split(key, 2)
        state = self.strategy.initialize(subkey)
        if self.start_mutations is not None:
            state = state.replace(mutations=self.start_mutations)

        if init_X is not None:
            temp = init_X.reshape((1, init_X.shape[0], init_X.shape[1]))
            new_archive = jnp.concatenate([temp, state.archive[1:, :, :]])
            state = state.replace(archive=new_archive)

        last_fitness = None
        best_fitness_avg = 100000
        last_best_fitness_avg = None

        # MUT_UPT_CNT = 10000 // self.popsize
        MUT_UPT_CNT = 1
        counter = 0

        for t in range(self.num_generations):
            self.key, ask_subkey, eval_subkey = jax.random.split(self.key, 3)
            # Produce new candidates
            ask_start = time.time()
            x, state = self.strategy.ask(ask_subkey, state)
            # if t <= 2:
            #     print(f't={t}) ask time = {time.time() - ask_start:.3f}')

            # Fitness of each candidate
            fit_start = time.time()
            fitness = fitness_fn(x)
            # if t <= 2:
            #     print(f't={t}) fit time = {time.time() - fit_start:.3f}')


            # Get next population
            state = self.strategy.tell(x, fitness, state)


            best_fitness = fitness.min()
            if best_fitness > best_fitness_avg:
                counter = counter + 1
                if counter >= MUT_UPT_CNT:
                    state = state.replace(mutations=(state.mutations + 1) // 2)
                    counter = 0

            # Early stop
            best_fitness_avg = min(best_fitness_avg, best_fitness)
            X_sync = state.best_member
            max_error = stat.priv_loss_inf(X_sync)

            # print(f'{t:03}) fitness = {fitness.min():.4f}, max_error={max_error:.4f}')
            if max_error < tolerance:
                if self.print_progress:
                    print(f'Stop early (1) at epoch {t}: max_error= {max_error:.4f} < {tolerance}.')
                break
            if t % self.stop_loss_time_window == 0 and t > 0:
                if last_best_fitness_avg is not None:
                    percent_change = jnp.abs(best_fitness_avg - last_best_fitness_avg) / last_best_fitness_avg
                    if percent_change < 0.001:
                        if self.print_progress:
                            print(f'Stop early (2) at epoch {t}:')
                        break

                last_best_fitness_avg = best_fitness_avg
                best_fitness_avg = 100000

            if last_fitness is None or best_fitness < last_fitness * 0.95 or t > self.num_generations-2 :
                if self.print_progress:
                    print(f'\tGeneration {t}, best_l2_fitness = {jnp.sqrt(best_fitness):.3f}, ', end=' ')
                    print(f'\ttime={time.time() -init_time:.3f}(s):', end='')
                    print(f'\t\tprivate max_error={max_error:.3f}', end='')
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
                 cross_rate: float = 0.1,
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

        self.cross_rate = cross_rate
        self.mate_vmap = jax.jit(jax.vmap(single_mate, in_axes=(0, 0, 0, None)))

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
        print(f'Debug strategy.ask(): This print should appear only once.')
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
        C = self.mate_vmap(rng_mate_split, A, B, self.cross_rate)


        rng_mutate = jax.random.split(rng_2, self.popsize // 2)
        A_mut = self.mutate_vmap(rng_mutate, A, state.mutations)
        x = jnp.concatenate((A_mut, C))
        return x, state


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
    rng: chex.PRNGKey, a: chex.Array, b: chex.Array, r: float
) -> chex.Array:
    """Only cross-over dims for x% of all dims."""
    n, d = a.shape
    # n, d = sync_data_shape

    X = a
    Y = b

    # rng1, rng2 = jax.random.split(rng, 2)
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
    cross_over_rate = r * jax.random.uniform(rng1, shape=(1,))

    idx = (jax.random.uniform(rng2, (n, )) > cross_over_rate).reshape((n, 1))
    X = jax.random.permutation(rng3, X, axis=0)
    Y = jax.random.permutation(rng4, Y, axis=0)

    XY = X * (1 - idx) + Y * idx
    cross_over_candidate = XY
    return cross_over_candidate

if __name__ == "__main__":
    # test_crossover()

    test_mutation()