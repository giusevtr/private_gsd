"""
To run parallelization on multiple cores set
XLA_FLAGS=--xla_force_host_platform_device_count=4
"""
import jax.numpy as jnp
from models import Generator
import time
from stats import Marginals, PrivateMarginalsState


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

"""
Implement crossover that is specific to synthetic data
"""
class SimpleGAforSyncData:
    def __init__(self, domain: Domain, population_size: int, elite_size: int, data_size: int, muta_rate: int, mate_rate: int):
        """Simple Genetic Algorithm For Synthetic Data Search Adapted from (Such et al., 2017)
        Reference: https://arxiv.org/abs/1712.06567
        Inspired by: https://github.com/hardmaru/estool/blob/master/es.py"""

        self.population_size = population_size
        self.elite_size = elite_size
        self.data_size = data_size

        self.domain = domain
        self.strategy_name = "SimpleGA"
        self.num_devices = jax.device_count()
        self.domain = domain

        mutate = get_mutation_fn(domain, mute_rate=muta_rate)
        mate_fn = get_mating_fn(mate_rate=mate_rate)

        self.mutate_vmap = jax.vmap(mutate, in_axes=(0, 0))
        self.mate_vmap = jax.vmap(mate_fn, in_axes=(0, 0, 0))

    @partial(jax.jit, static_argnums=(0,))
    def initialize(
        self, rng: chex.PRNGKey,
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        # Initialize strategy based on strategy-specific initialize method
        state = self.initialize_strategy(rng)
        return state

    def initialize_strategy(self, rng: chex.PRNGKey) -> EvoState:
        """`initialize` the differential evolution strategy."""
        initialization = initialize_population(rng, self.elite_size, self.domain, self.data_size).astype(jnp.float32)

        state = EvoState(
            mean=initialization.mean(axis=0),
            archive=initialization,
            fitness=jnp.zeros(self.elite_size) + jnp.finfo(jnp.float32).max,
            best_member=initialization[0],
        )
        return state

    @partial(jax.jit, static_argnums=(0,))
    def ask(
        self,
        rng: chex.PRNGKey,
        state: EvoState,
            # popsize: int,
            # elite_popsize: int,
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        x, state = self.ask_strategy(rng, state)
        return x, state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState
    ) -> Tuple[chex.Array, EvoState]:
        rng, rng_a, rng_b, rng_mate, rng_2 = jax.random.split(rng, 5)
        elite_ids = jnp.arange(self.elite_size)
        pop_size = self.population_size // 2

        idx_a = jax.random.choice(rng_a, elite_ids, (pop_size,))
        idx_b = jax.random.choice(rng_b, elite_ids, (pop_size,))
        A = state.archive[idx_a]
        B = state.archive[idx_b]

        rng_mate_split = jax.random.split(rng_mate, pop_size)
        C = self.mate_vmap(rng_mate_split, A, B)


        x = jnp.concatenate((A, C))
        rng_mutate = jax.random.split(rng_2, self.population_size)
        x = self.mutate_vmap(rng_mutate, x)
        return x, state


    @partial(jax.jit, static_argnums=(0,))
    def tell(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
            # elite_popsize
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
            # elite_popsize: int
    ) -> EvoState:
        """
        `tell` update to ES state.
        If fitness of y <= fitness of x -> replace in population.
        """
        # Combine current elite and recent generation info
        fitness = jnp.concatenate([fitness, state.fitness])
        solution = jnp.concatenate([x, state.archive])
        # Select top elite from total archive info
        idx = jnp.argsort(fitness)[0: self.elite_size]

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
    d = len(domain.attrs)
    pop = Dataset.synthetic_jax_rng(domain, pop_size * data_size, rng)
    initialization = pop.reshape((pop_size, data_size, d))

    # for s in range(pop_size):
    #     rng, rng_sub = jax.random.split(rng)
    #     X = Dataset.synthetic_jax_rng(domain, data_size, rng_sub)
    #     temp.append(X)
    #
    # initialization = jnp.array(temp)
    return initialization




######################################################################
######################################################################
######################################################################
######################################################################
######################################################################


# @dataclass
class PrivGA(Generator):

    def __init__(self,
                 num_generations,
                 stop_loss_time_window,
                 print_progress,
                 strategy: SimpleGAforSyncData,
                    time_limit: float = None,
                 ):
        self.domain = strategy.domain
        self.data_size = strategy.data_size
        self.num_generations = num_generations
        self.stop_loss_time_window = stop_loss_time_window
        self.print_progress = print_progress
        self.strategy = strategy
        self.time_limit = time_limit

    def __str__(self):
        return f'PrivGA'

    def fit(self, key, stat: PrivateMarginalsState, init_X=None, tolerance: float=0.0):
        """
        Minimize error between real_stats and sync_stats
        """
        key, key_sub = jax.random.split(key, 2)

        # key = jax.random.PRNGKey(seed)
        init_time = time.time()
        num_devices = jax.device_count()
        if num_devices > 1:
            print(f'************ {num_devices}  devices found. Using parallelization. ************')

        # FITNESS
        fitness_fn = jax.jit(stat.priv_loss_l2_vmap_jit)

        self.key, subkey = jax.random.split(key, 2)
        state = self.strategy.initialize(subkey)

        if init_X is not None:
            temp = init_X.reshape((1, init_X.shape[0], init_X.shape[1]))
            new_archive = jnp.concatenate([temp, state.archive[1:, :, :]])
            state = state.replace(archive=new_archive)

        last_fitness = None
        best_fitness_total = 100000
        smooth_loss_sum = 0
        last_loss = None
        start_time: float = 0.0
        self.early_stop_init()
        total_best_fitness = 1000000

        ask_time = 0
        fit_time = 0
        tell_time = 0
        for t in range(self.num_generations):
            self.key, ask_subkey, eval_subkey = jax.random.split(self.key, 3)
            # Produce new candidates
            t0 = time.time()
            x, state = self.strategy.ask(ask_subkey, state)
            ask_time += time.time() - t0


            # Fitness of each candidate
            t0 = time.time()
            fitness = fitness_fn(x)
            fit_time += time.time() - t0


            # Get next population
            t0 = time.time()
            state = self.strategy.tell(x, fitness, state)
            tell_time += time.time() - t0

            best_fitness = fitness.min()

            # Early stop
            best_fitness_total = min(best_fitness_total, best_fitness)

            stop_early = self.early_stop(best_fitness_total)

            if last_fitness is None or best_fitness < last_fitness * 0.99 or t > self.num_generations-2 or stop_early:
                if self.print_progress:
                    X_sync = state.best_member
                    print(f'\tGeneration {t:05}, best_l2_fitness = {best_fitness:.6f}, ', end=' ')
                    print(f'\t\tprivate (max/l2) error={stat.priv_loss_inf(X_sync):.5f}/{stat.priv_loss_l2(X_sync):.7f}', end='')
                    print(f'\t\ttrue (max/l2) error={stat.true_loss_inf(X_sync):.5f}/{stat.true_loss_l2(X_sync):.7f}', end='')
                    print(f'\ttime={time.time() -init_time:.4f}(s):', end='')
                    print(f'\task_time={ask_time:.4f}(s), fit_time={fit_time:.4f}(s), tell_time={tell_time:.4f}', end='')
                    print()
                    if stop_early: print(f'\tStop early at {t}')

                last_fitness = best_fitness

            if stop_early: break

        X_sync = state.best_member
        sync_dataset = Dataset.from_numpy_to_dataset(self.domain, X_sync)
        if self.print_progress:
            print(f'\t\tFinal private max_error={stat.priv_loss_inf(X_sync):.3f}, private l2_error={stat.priv_loss_l2(X_sync):.6f},', end='\n')

        return sync_dataset



def get_mutation_fn(domain: Domain, mute_rate: int):

    def mutate(rng: chex.PRNGKey, X: chex.Array) -> chex.Array:
        n, d = X.shape
        rng1, rng2, rng3 = jax.random.split(rng, 3)
        initialization = Dataset.synthetic_jax_rng(domain, mute_rate, rng1)
        mut_rows = jax.random.randint(rng2, minval=0, maxval=n, shape=(mute_rate, ))
        # mut_rows = jax.random.choice(rng2, n, shape=(mutations, ), replace=False)  # THIS IS SLOWER THAN randint
        mut_col = jax.random.randint(rng3, minval=0, maxval=d, shape=(mute_rate, ))
        values = initialization[jnp.arange(mute_rate), mut_col]
        X = X.at[mut_rows, mut_col].set(values)

        # Eval mutate rows.
        return X

    return mutate


def get_mating_fn(mate_rate: int):
    def single_mate(rng: chex.PRNGKey, X: chex.Array, Y: chex.Array) -> chex.Array:
        n_X, d = X.shape
        n_Y, d = X.shape
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

        rows_X = jax.random.randint(rng1, minval=0,  maxval=n_X, shape=(mate_rate, ))
        rows_Y = jax.random.randint(rng2, minval=0,  maxval=n_Y, shape=(mate_rate, ))
        XY = X.at[rows_X].set(Y[rows_Y, :])
        return XY
    return single_mate


######################################################################
######################################################################
######################################################################
######################################################################
######################################################################


def test_mutation():

    domain = Domain(['A', 'B', 'C'], [10, 10, 1])
    mutate = get_mutation_fn(domain, mute_rate=3)
    rng = jax.random.PRNGKey(2)
    x = Dataset.synthetic_jax_rng(domain, 4, rng)
    rng, rng_sub = jax.random.split(rng)
    x2 = mutate(rng_sub, x)

    print('x =')
    print(x)
    print('x2=')
    print(x2)


    print(f'Runtime test:')
    SYNC_SIZE = 5000

    domain_large = Domain([f'f{i}' for i in range(30)], [2 for _ in range(15)] + [1 for _ in range(15)])
    mutate2 = get_mutation_fn(domain_large, mute_rate=100)
    mutate2_jit = jax.jit(mutate2)
    x = Dataset.synthetic_jax_rng(domain_large, SYNC_SIZE, rng)
    rng = jax.random.PRNGKey(5)
    for t in range(4):
        rng, rng_sub = jax.random.split(rng)
        stime = time.time()
        x_mutated = mutate2_jit(rng_sub, x)
        x_mutated.block_until_ready()
        print(f'{t}) Elapsed time: {time.time() - stime:.4f}')


def test_mating():

    domain = Domain(['A', 'B', 'C'], [10, 10, 1])
    rng = jax.random.PRNGKey(3)
    rng, rng1, rng2, rng_mate = jax.random.split(rng, 4)
    X = Dataset.synthetic_jax_rng(domain, 5, rng1)
    Y = Dataset.synthetic_jax_rng(domain, 5, rng2)
    single_mate_small = get_mating_fn(mate_rate=2)

    x_mate = single_mate_small(rng_mate, X, Y)

    print('X =')
    print(X)
    print('Y =')
    print(Y)

    print(f'x_mate:')
    print(x_mate)



    print(f'Runtime test:')

    SYNC_SIZE = 5000
    mate_rate = 500

    #
    domain_large = Domain([f'f{i}' for i in range(30)], [2 for _ in range(15)] + [1 for _ in range(15)])
    single_mate = get_mating_fn(mate_rate=mate_rate)
    mate_jit = jax.jit(single_mate)

    rng = jax.random.PRNGKey(5)
    rng, rng1, rng2 = jax.random.split(rng, 3)
    X = Dataset.synthetic_jax_rng(domain_large, SYNC_SIZE, rng1)
    Y = Dataset.synthetic_jax_rng(domain_large, SYNC_SIZE, rng2)
    for t in range(4):
        rng, rng_sub = jax.random.split(rng)
        stime = time.time()
        XY = mate_jit(rng_sub, X, Y)
        XY.block_until_ready()
        print(f'{t}) Elapsed time: {time.time() - stime:.4f}')


# @timeit
def test_jit_ask():
    rounds = 10
    d = 20
    k = 1
    domain = Domain([f'A {i}' for i in range(d)], [3 for _ in range(d)])
    print(f'Test jit(ask) with {rounds} rounds. d={d}, k={k}')

    strategy = SimpleGAforSyncData(domain, population_size=200, elite_size=10, data_size=2000,
                                   muta_rate=1,
                                   mate_rate=0)
    stime = time.time()
    key = jax.random.PRNGKey(0)
    state = strategy.initialize(rng=key)
    print(f'Initialize elapsed time {time.time() - stime:.3f}s')

    print()
    for r in range(rounds):
        stime = time.time()
        x, state = strategy.ask(key, state)
        x.block_until_ready()

        # if r <= 3 or r == rounds - 1:
        print(f'{r:>3}) ask() elapsed time {time.time() - stime:.5f}')
        print()


if __name__ == "__main__":

    # test_crossover()

    # test_mutation()
    # test_mating()
    test_jit_ask()