"""
To run parallelization on multiple cores set
XLA_FLAGS=--xla_force_host_platform_device_count=4
"""
import jax.numpy as jnp
from models import Generator
import time
from stats import Marginals, PrivateMarginalsState
from typing import Callable


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
    archive_stats: chex.Array
    fitness: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0

"""
Implement crossover that is specific to synthetic data
"""
class SimpleGAforSyncDataFast:
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
        self.muta_rate = muta_rate

        self.mate_rate = mate_rate

        mutate = get_mutation_fn(domain, mute_rate=muta_rate)
        mate_fn = get_mating_fn(mate_rate=mate_rate)

        self.mutate_vmap = jax.vmap(mutate, in_axes=(0, 0))
        self.mate_vmap = jax.vmap(mate_fn, in_axes=(0, 0, 0))

    def initialize(
        self, rng: chex.PRNGKey, eval_stats_vmap: Callable
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        # Initialize strategy based on strategy-specific initialize method
        # state = self.initialize_strategy(rng)
        # N, D = state.archive[0].shape
        # elite_stats = eval_stats_vmap(state.archive)  # Get stats of each archive dataset
        # state = state.replace(elite_stats=elite_stats)
        # return state
        init_x = self.initialize_population(rng)
        init_a = self.data_size * eval_stats_vmap(init_x)
        # print("check,shape:",init_a.shape)
        state = EvoState(
            mean=init_x.mean(axis=0),
            archive=init_x,
            archive_stats=init_a,
            fitness=jnp.zeros(self.elite_size) + jnp.finfo(jnp.float32).max,
            best_member=init_x[0],
        )
        return state

    # # @partial(jax.jit, static_argnums=(0,))
    # def initialize_strategy(self, rng: chex.PRNGKey) -> EvoState:
    #     """`initialize` the differential evolution strategy."""
    #     initialization = self.initialize_population(rng).astype(jnp.float32)
    #
    #     state = EvoState(
    #         mean=initialization.mean(axis=0),
    #         archive=initialization,
    #         archive_row_answers=jnp.zeros((self.elite_size, 1)),
    #         fitness=jnp.zeros(self.elite_size) + jnp.finfo(jnp.float32).max,
    #         best_member=initialization[0],
    #     )
    #     return state

    @partial(jax.jit, static_argnums=(0,))
    def initialize_population(self, rng: chex.PRNGKey):
        d = len(self.domain.attrs)
        pop = Dataset.synthetic_jax_rng(self.domain, self.elite_size * self.data_size, rng)
        initialization = pop.reshape((self.elite_size, self.data_size, d))
        return initialization

    def ask_strategy(self, rng: chex.PRNGKey, eval_stats_vmap, state):
        x_mutated, old_rows, new_rows, idx_elite = self.ask_mutate_help(rng, state)
        # print(f'debug1: {time.time() - stime}')
        # print("old_rows:",old_rows.shape)
        pop_ind = jnp.arange(self.population_size)
        old_stats = eval_stats_vmap(old_rows)
        # print("old_stats:",old_stats.shape)
        new_stats = eval_stats_vmap(new_rows)

        # Update stats
        a = state.archive_stats[idx_elite]  # With corresponding statistics
        # print("a:",a.shape)
        a_updated = a.at[pop_ind].add(new_stats - old_stats)

        return x_mutated, a_updated, state

    @partial(jax.jit, static_argnums=(0, ))
    def ask_mutate_help(
            self, rng: chex.PRNGKey, state
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:

        # rng, rng_mutate2 = jax.random.split(rng)
        # mut_row_indices, mut_col_indices, new_values = self.ask_mutate_x_strategy_jit(rng_mutate2)

        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        D = len(self.domain.attrs)
        initialization = Dataset.synthetic_jax_rng(self.domain, self.population_size, rng1)
        # Row to be mutation for each new population member
        mut_row_indices = jax.random.randint(rng2, minval=0, maxval=self.data_size, shape=(self.population_size,))
        mut_col_indices = jax.random.randint(rng3, minval=0, maxval=D, shape=(self.population_size,))
        new_values = initialization[jnp.arange(self.population_size), mut_col_indices]

        rng, rng_elite_set = jax.random.split(rng)
        idx_elite = jax.random.randint(rng_elite_set, minval=0, maxval=self.elite_size,  shape=(self.population_size, ))
        x = state.archive[idx_elite]  # Population derived from elite set

        N, D = x[0].shape
        pop_ind = jnp.arange(self.population_size)
        old_rows = x[pop_ind, mut_row_indices, :].reshape((self.population_size, 1, D))

        x_mutated = x.at[pop_ind, mut_row_indices, mut_col_indices].set(new_values)
        new_rows = x_mutated[pop_ind, mut_row_indices, :].reshape((self.population_size, 1, D))

        return x_mutated, old_rows, new_rows, idx_elite





    # @partial(jax.jit, static_argnums=(0, ))
    # def ask_mate(
    #     self, rng: chex.PRNGKey, state: EvoState
    # ) -> Tuple[chex.Array, chex.Array, EvoState]:
    #     return self.ask_mate_strategy(rng, state)

    # def ask_mate_strategy(
    #     self, rng: chex.PRNGKey, state: EvoState
    # ) -> Tuple[chex.Array, chex.Array, EvoState]:
    #     rng, rng_a, rng_b, rng_mate, rng_2 = jax.random.split(rng, 5)
    #     elite_ids = jnp.arange(self.elite_size)
    #     N, D = state.archive[0].shape
    #     idx_x1 = jax.random.choice(rng_a, elite_ids, (self.population_size // 2,))
    #     idx_x2 = jax.random.choice(rng_b, elite_ids, (self.population_size // 2,))
    #
    #     rgn, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)
    #     rows_A = jax.random.randint(rng1, minval=0,  maxval=N * self.elite_size, shape=(self.mate_rate * self.elite_size, ))
    #     rows_B = jax.random.randint(rng2, minval=0,  maxval=N * self.elite_size, shape=(self.mate_rate * self.elite_size, ))
    #
    #     idx_x1 = idx_x1.at[rows_A].set(idx_x2[rows_B])
    #
    #
    #     C = A.at[rows_A].set(B[rows_B, :])
    #     C_answers = A_answers.at[rows_A].set(B_answers[rows_B, :])
    #     x = jnp.concatenate((A, C))
    #     a = jnp.concatenate((A_answers, C_answers))
    #     x = x.reshape((self.population_size, N, D))
    #     a = a.reshape((self.population_size, N, -1))
    #     return x, a, state
    # def ask_mate_strategy(
    #     self, rng: chex.PRNGKey, state: EvoState
    # ) -> Tuple[chex.Array, chex.Array, EvoState]:
    #     rng, rng_a, rng_b, rng_mate, rng_2 = jax.random.split(rng, 5)
    #     elite_ids = jnp.arange(self.elite_size)
    #     N, D = state.archive[0].shape
    #     idx_a = jax.random.choice(rng_a, elite_ids, (self.population_size // 2,))
    #     idx_b = jax.random.choice(rng_b, elite_ids, (self.population_size // 2,))
    #     A = state.archive[idx_a].reshape(-1, D)
    #     A_answers = state.archive_row_answers[idx_a]
    #     B = state.archive[idx_b].reshape(-1, D)
    #     B_answers = state.archive_row_answers[idx_b]
    #     rgn, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)
    #     rows_A = jax.random.randint(rng1, minval=0,  maxval=N * self.elite_size, shape=(self.mate_rate * self.elite_size, ))
    #     rows_B = jax.random.randint(rng2, minval=0,  maxval=N * self.elite_size, shape=(self.mate_rate * self.elite_size, ))
    #     C = A.at[rows_A].set(B[rows_B, :])
    #     C_answers = A_answers.at[rows_A].set(B_answers[rows_B, :])
    #     x = jnp.concatenate((A, C))
    #     a = jnp.concatenate((A_answers, C_answers))
    #     x = x.reshape((self.population_size, N, D))
    #     a = a.reshape((self.population_size, N, -1))
    #     return x, a, state


    @partial(jax.jit, static_argnums=(0,))
    def tell(
        self,
        x: chex.Array,
        a: chex.Array,
        fitness: chex.Array,
        state: EvoState,
            # elite_popsize
    ) -> EvoState:
        """`tell` performance data for strategy state update."""

        # Update the search state based on strategy-specific update
        state = self.tell_strategy(x, a, fitness, state)

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
        a: chex.Array,
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
        stats = jnp.concatenate([a, state.archive_stats])
        # Select top elite from total archive info
        idx = jnp.argsort(fitness)[0: self.elite_size]

        ## MODIFICATION: Select random survivors
        fitness = fitness[idx]
        archive = solution[idx]
        archive_stats = stats[idx]
        # Update mutation epsilon - multiplicative decay

        # Keep mean across stored archive around for evaluation protocol
        mean = archive.mean(axis=0)
        return state.replace(
            fitness=fitness, archive=archive, archive_stats=archive_stats, mean=mean
        )







######################################################################
######################################################################
######################################################################
######################################################################
######################################################################


# @dataclass
class PrivGAfast(Generator):

    def __init__(self,
                 num_generations,
                 stop_loss_time_window,
                 print_progress,
                 strategy: SimpleGAforSyncDataFast,
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
        return f'PrivGAfast'

    def fit(self, key, priv_stat_module: PrivateMarginalsState, init_X=None, tolerance: float=0.0):
        """
        Minimize error between real_stats and sync_stats
        """
        key, key_sub = jax.random.split(key, 2)

        # key = jax.random.PRNGKey(seed)
        init_time = time.time()
        num_devices = jax.device_count()
        if num_devices > 1:
            print(f'************ {num_devices}  devices found. Using parallelization. ************')

        get_stats_vmap = lambda X: priv_stat_module.get_stats(X)
        get_stats_jax_vmap = jax.vmap(get_stats_vmap, in_axes=(0,))

        self.key, subkey = jax.random.split(key, 2)
        state = self.strategy.initialize(subkey, get_stats_jax_vmap)


        fitness_fn = lambda sync_stat : jnp.linalg.norm(priv_stat_module.get_true_stats() - sync_stat, ord=2)
        # FITNESS
        fitness_vmap_fn = jax.vmap(fitness_fn, in_axes=(0, ))

        # if init_X is not None:
        #     temp = init_X.reshape((1, init_X.shape[0], init_X.shape[1]))
        #     new_archive = jnp.concatenate([temp, state.archive[1:, :, :]])
        #     state = state.replace(archive=new_archive)

        last_fitness = None
        best_fitness_total = 100000
        self.early_stop_init()

        for t in range(self.num_generations):
            self.key, ask_subkey, eval_subkey = jax.random.split(self.key, 3)
            # Produce new candidates
            stime = time.time()
            x, a, state = self.strategy.ask_strategy(ask_subkey, get_stats_jax_vmap, state)
            # print(f'ask.time = {time.time() - stime:.5f}')

            stime = time.time()
            # Fitness of each candidate
            fitness = fitness_vmap_fn(a / self.data_size)
            # print(f'fit.time = {time.time() - stime:.5f}')

            # Get next population
            stime = time.time()
            state = self.strategy.tell(x, a, fitness, state)
            # print(f'tell.time = {time.time() - stime:.5f}')

            best_fitness = fitness.min()

            # Early stop
            best_fitness_total = min(best_fitness_total, best_fitness)


            if self.early_stop(best_fitness_total):
                if self.print_progress:
                    print(f'\tStop early at {t}')
                break

            X_sync = state.best_member
            max_error = priv_stat_module.priv_loss_inf(X_sync)
            if max_error < tolerance:
                if self.print_progress:
                    print(f'\tTolerance hit at t={t}')
                break

            if last_fitness is None or best_fitness < last_fitness * 0.95 or t > self.num_generations-2 :
                if self.print_progress:
                    print(f'\tGeneration {t:05}, best_l2_fitness = {jnp.sqrt(best_fitness):.6f}, ', end=' ')
                    print(f'\t\tprivate (max/l2) error={priv_stat_module.priv_loss_inf(X_sync):.5f}/{priv_stat_module.priv_loss_l2(X_sync):.7f}', end='')
                    print(f'\t\ttrue (max/l2) error={priv_stat_module.true_loss_inf(X_sync):.5f}/{priv_stat_module.true_loss_l2(X_sync):.7f}', end='')
                    print(f'\ttime={time.time() -init_time:.7f}(s):', end='')
                    print()
                last_fitness = best_fitness

            if t == 0:
                start_time = time.time()
        X_sync = state.best_member
        sync_dataset = Dataset.from_numpy_to_dataset(self.domain, X_sync)
        if self.print_progress:
            print(f'\t\tFinal private max_error={priv_stat_module.priv_loss_inf(X_sync):.3f}, private l2_error={priv_stat_module.priv_loss_l2(X_sync):.6f},', end='\n')

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


from stats import Marginals
# @timeit
def test_jit_ask(rounds):
    d = 30
    domain = Domain([f'A {i}' for i in range(d)], [3 for _ in range(d)])
    data = Dataset.synthetic(domain, N=10, seed=0)

    domain = data.domain
    print(f'Test jit(ask) with {rounds} rounds.')

    marginals = Marginals.get_all_kway_combinations(domain, k=1, bins=[2])
    marginals.fit(data)

    eval_stats_vmap = lambda x: marginals.get_stats_jax_vmap(x)
    eval_stats_vmap = jax.jit(eval_stats_vmap)

    strategy = SimpleGAforSyncDataFast(domain, population_size=200, elite_size=30, data_size=2000,
                                   muta_rate=1,
                                   mate_rate=1)
    stime = time.time()
    key = jax.random.PRNGKey(0)

    #Initial population and statistics
    stime = time.time()
    init_x = strategy.initialize_population(key)
    stime = time.time()
    init_a = strategy.data_size * eval_stats_vmap(init_x)
    state = EvoState(
        mean=init_x.mean(axis=0),
        archive=init_x,
        archive_stats=init_a,
        fitness=jnp.zeros(strategy.elite_size) + jnp.finfo(jnp.float32).max,
        best_member=init_x[0],
    )


    # state = strategy.initialize(rng=key)
    print(f'Initialize elapsed time {time.time() - stime:.3f}s')

    print()
    for r in range(rounds):
        stime = time.time()
        # x, a, state = stregy.ask_mate(key, state)
        x, a, state = strategy.ask_strategy(key, eval_stats_vmap, state)


        x.block_until_ready()

        # fitness = jnp.zeros(200)
        # state = strategy.tell(x, a, fitness, state)
        if r <= 3 or r == rounds - 1:
            print(f'{r:>3}) Jitted elapsed time {time.time() - stime:.3f}')


if __name__ == "__main__":

    # test_crossover()

    # test_mutation()
    # test_mating()
    test_jit_ask(rounds=10)