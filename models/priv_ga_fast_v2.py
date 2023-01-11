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

def timer(last_time=None, msg=None):
    now = time.time()
    if msg is not None and last_time is not None:
        print(f'{msg} {now - last_time:.5f}')
    return now

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
    def __init__(self, domain: Domain, population_size: int, elite_size: int, data_size: int, muta_rate: int, mate_rate: int,
                 debugging=False):
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


        self.debugging = debugging

        sample_data_rows = lambda key: jax.random.choice(key, self.data_size, replace=False, shape=(self.mate_rate,))
        self.sample_population_rows = jax.jit(jax.vmap(sample_data_rows, in_axes=(0, )))

    def initialize(
        self, rng: chex.PRNGKey, get_stats_vmap: Callable
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        init_x = self.initialize_population(rng)
        init_a = self.data_size * get_stats_vmap(init_x)
        state = EvoState(
            mean=init_x.mean(axis=0),
            archive=init_x,
            archive_stats=init_a,
            fitness=jnp.zeros(self.elite_size) + jnp.finfo(jnp.float32).max,
            best_member=init_x[0],
        )

        rng1, rng2 = jax.random.split(rng, 2)
        # keys1 = jax.random.split(rng1, self.population_size//2)
        # self.mate_rows_idx_1 = self.sample_population_rows(keys1)

        mutate = get_mutation_fn(self.domain, muta_rate=self.muta_rate, get_stats_vmap=get_stats_vmap)

        random_numbers = jax.random.permutation(rng1, self.data_size, independent=True)
        mate_fn = get_mating_fn(mate_rate=self.mate_rate, get_stats_vmap=get_stats_vmap, random_numbers=random_numbers)

        self.mutate_vmap = jax.jit(jax.vmap(mutate, in_axes=(0, None)))
        self.mate_vmap = jax.jit(jax.vmap(mate_fn, in_axes=(0, None)))

        return state

    # @partial(jax.jit, static_argnums=(0,))
    def initialize_strategy(self, rng: chex.PRNGKey) -> EvoState:
        """`initialize` the differential evolution strategy."""
        initialization = self.initialize_population(rng).astype(jnp.float32)

        state = EvoState(
            mean=initialization.mean(axis=0),
            archive=initialization,
            archive_row_answers=jnp.zeros((self.elite_size, 1)),
            fitness=jnp.zeros(self.elite_size) + jnp.finfo(jnp.float32).max,
            best_member=initialization[0],
        )
        return state

    @partial(jax.jit, static_argnums=(0,))
    def initialize_population(self, rng: chex.PRNGKey):
        d = len(self.domain.attrs)
        pop = Dataset.synthetic_jax_rng(self.domain, self.elite_size * self.data_size, rng)
        initialization = pop.reshape((self.elite_size, self.data_size, d))
        return initialization



    @partial(jax.jit, static_argnums=(0,))
    def ask(self, rng: chex.PRNGKey, state: EvoState):
        rng_mut, rng_mate = jax.random.split(rng, 2)
        x_mut, a_mut = self.ask_mutate_strategy(rng_mut, state)
        # return x_mut, x_mut, state
        x_mat, a_mat = self.ask_mate_strategy(rng_mate, state)
        # return x_mat, a_mat, state
        x = jnp.concatenate((x_mut, x_mat))
        a = jnp.concatenate((a_mut, a_mat))
        return x, a, state


    def ask_mutate_strategy(self, rng: chex.PRNGKey, state: EvoState)-> Tuple[chex.Array, chex.Array]:
        pop_size = self.population_size // 2
        # pop_size = self.population_size
        rng_mutate = jax.random.split(rng, pop_size)
        x_mutated, a_updated = self.mutate_vmap(rng_mutate, state)
        return x_mutated, a_updated

    def ask_mate_strategy(self, rng: chex.PRNGKey, state: EvoState) -> Tuple[chex.Array, chex.Array]:
        pop_size = self.population_size // 2
        # pop_size = self.population_size
        # x, a, removed_rows, added_rows, state = self.ask_mate_help_strategy(rng, state)
        rng_mate = jax.random.split(rng, pop_size)
        x, a = self.mate_vmap(rng_mate, state)

        return x, a


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
        fitness = jnp.concatenate([fitness, state.fitness])
        solution = jnp.concatenate([x, state.archive])
        stats = jnp.concatenate([a, state.archive_stats])
        idx = jnp.argsort(fitness)[0: self.elite_size]

        fitness = fitness[idx]
        archive = solution[idx]
        archive_stats = stats[idx]

        mean = archive.mean(axis=0)
        return state.replace(
            fitness=fitness, archive=archive, archive_stats=archive_stats, mean=mean
        )



def get_mutation_fn(domain: Domain, muta_rate: int, get_stats_vmap: Callable):

    def mutate(rng: chex.PRNGKey, state: EvoState) -> [chex.Array, chex.Array]:

        elite_size = state.archive.shape[0]
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
        i = jax.random.randint(rng1, minval=0, maxval=elite_size, shape=(1, ))
        X = state.archive[i]
        Stats = state.archive_stats[i]

        X = X[0]
        A = Stats[0]
        n, d = X.shape
        initialization = Dataset.synthetic_jax_rng(domain, muta_rate, rng2)
        mut_rows = jax.random.randint(rng2, minval=0, maxval=n, shape=(muta_rate,))
        # mut_rows = jax.random.choice(rng2, n, shape=(mutations, ), replace=False)  # THIS IS SLOWER THAN randint
        mut_col = jax.random.randint(rng3, minval=0, maxval=d, shape=(muta_rate,))
        values = initialization[jnp.arange(muta_rate), mut_col]
        removed_rows = X[mut_rows, :]
        X_mut = X.at[mut_rows, mut_col].set(values)
        added_rows = X_mut[mut_rows, :]

        removed_stats = get_stats_vmap(removed_rows).squeeze()
        added_stats = get_stats_vmap(added_rows).squeeze()
        a_updated = A + added_stats - removed_stats
        return X_mut, a_updated

    return mutate



def get_mating_fn(mate_rate: int, get_stats_vmap: Callable, random_numbers):


    # random_numbers = jax.random.permutation(rng1, jnp.arange(n), independent=True)[:mate_rate]
    def single_mate(rng: chex.PRNGKey, state: EvoState) -> \
            [chex.Array, chex.Array]:
        elite_size = state.archive.shape[0]

        rng, rng_eli_1, rng_eli_2 = jax.random.split(rng, 3)

        idx_elite_1 = jax.random.randint(rng_eli_1, minval=0, maxval=elite_size,  shape=(1,))
        idx_elite_2 = jax.random.randint(rng_eli_2, minval=0, maxval=elite_size,  shape=(1,))

        X1 = state.archive[idx_elite_1][0]
        X2 = state.archive[idx_elite_2][0]

        init_stats = state.archive_stats[idx_elite_1][0]

        n, d = X1.shape
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        temp = jax.random.randint(rng1, minval=0, maxval=n - mate_rate, shape=(1, ))

        temp_id = jnp.arange(mate_rate) + temp
        remove_rows_idx = random_numbers[temp_id]


        # remove_rows_idx = jax.random.permutation(rng1, jnp.arange(n), independent=True)[:mate_rate]
        # remove_rows_idx = jax.random.choice(rng1, n, replace=False, shape=(mate_rate, ))
        # remove_rows_idx = jax.random.randint(rng1, minval=0,  maxval=n, shape=(mate_rate, ))
        add_rows_idx = jax.random.randint(rng2, minval=0,  maxval=n, shape=(mate_rate, ))
        X = X1.at[remove_rows_idx].set(X2[add_rows_idx, :])
        removed_rows = X1[remove_rows_idx, :].reshape((1, mate_rate, d))
        added_rows = X2[add_rows_idx, :].reshape((1, mate_rate, d))

        removed_stats = mate_rate * get_stats_vmap(removed_rows).squeeze()
        added_stats = mate_rate * get_stats_vmap(added_rows).squeeze()
        stats = init_stats + added_stats - removed_stats
        return X, stats



        # return X, stats

    return single_mate




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

        init_time = time.time()
        # key = jax.random.PRNGKey(seed)
        num_devices = jax.device_count()
        if num_devices > 1:
            print(f'************ {num_devices}  devices found. Using parallelization. ************')

        get_stats_vmap = lambda X: priv_stat_module.get_stats(X)
        get_stats_jax_vmap = jax.vmap(get_stats_vmap, in_axes=(0,))
        get_stats_jax_vmap = jax.jit(get_stats_jax_vmap)

        self.key, subkey = jax.random.split(key, 2)
        state = self.strategy.initialize(subkey, get_stats_jax_vmap)

        priv_stats = priv_stat_module.get_priv_stats()
        # FITNESS
        fitness_fn = lambda sync_stat: jnp.linalg.norm(priv_stats - sync_stat, ord=2)
        fitness_vmap_fn = jax.vmap(fitness_fn, in_axes=(0, ))
        fitness_vmap_fn = jax.jit(fitness_vmap_fn)

        # if init_X is not None:
        #     temp = init_X.reshape((1, init_X.shape[0], init_X.shape[1]))
        #     new_archive = jnp.concatenate([temp, state.archive[1:, :, :]])
        #     state = state.replace(archive=new_archive)

        last_fitness = None
        best_fitness_total = 100000
        self.early_stop_init()

        @jax.jit
        def concat_x(x1, x2):
            return jnp.concatenate([x1, x2])
        @jax.jit
        def concat_a(a1, a2):
            return jnp.concatenate([a1, a2])

        ask_muta_time = 0
        ask_mate_time = 0
        fit_time = 0
        tell_time = 0
        for t in range(self.num_generations):
            self.key, mutate_subkey, mate_subkey = jax.random.split(self.key, 3)
            # Produce new candidates
            t0 = timer()
            x, a, state = self.strategy.ask(mutate_subkey, state)
            ask_muta_time += timer() - t0

            stime = time.time()
            # Fitness of each candidate
            fitness = fitness_vmap_fn(a / self.data_size)
            fit_time += time.time() - stime

            # Get next population
            t0 = timer()
            state = self.strategy.tell(x, a, fitness, state)
            tell_time += timer() - t0

            best_fitness = fitness.min()

            # Early stop
            best_fitness_total = min(best_fitness_total, best_fitness)

            if self.early_stop(best_fitness_total):
                if self.print_progress:
                    print(f'\tStop early at {t}')
                break

            if last_fitness is None or best_fitness < last_fitness * 0.95 or t > self.num_generations-2 :
                if self.print_progress:
                    X_sync = state.best_member
                    print(f'\tGeneration {t:05}, best_l2_fitness = {jnp.sqrt(best_fitness):.6f}, ', end=' ')
                    print(f'\t\tprivate (max/l2) error={priv_stat_module.priv_loss_inf(X_sync):.5f}/{priv_stat_module.priv_loss_l2(X_sync):.7f}', end='')
                    print(f'\t\ttrue (max/l2) error={priv_stat_module.true_loss_inf(X_sync):.5f}/{priv_stat_module.true_loss_l2(X_sync):.7f}', end='')
                    print(f'\ttime={timer() -init_time:.4f}(s):', end='')
                    print(f'\tmuta_t={ask_muta_time:.4f}(s), mate_t={ask_mate_time:.4f}(s)'
                          f'   fit_t={fit_time:.4f}(s), tell_t={tell_time:.4f}', end='')
                    print()
                last_fitness = best_fitness

        X_sync = state.best_member
        sync_dataset = Dataset.from_numpy_to_dataset(self.domain, X_sync)
        if self.print_progress:
            print(f'\t\tFinal private max_error={priv_stat_module.priv_loss_inf(X_sync):.3f}, private l2_error={priv_stat_module.priv_loss_l2(X_sync):.6f},', end='\n')

        return sync_dataset




######################################################################
######################################################################
######################################################################
######################################################################
######################################################################


def test_mutation_fn():
    domain = Domain(['A', 'B', 'C'], [10, 10, 1])
    muta_fn = get_mutation_fn(domain, muta_rate=1)
    muta_vamp = jax.vmap(muta_fn, in_axes=(0, 0))

    state = EvoState()
    key = jax.random.PRNGKey(0)

    pop = jnp.array([
                    [[0, 0, 0.5],
                    [0, 0, 0.9],
                     [1, 0, 0.9],
                    [0, 1, 0.9]],
                [[0, 0, 0.5],
                 [0, 0, 0.9],
                 [0, 0, 0.9],
                 [0, 1, 0.9]],
            ])

    pop_size = pop.shape[0]

    keys = jax.random.split(key, pop_size)
    new_pop, removed_rows, added_rows = muta_vamp(keys, pop)

    print(f'new_pop={new_pop.shape}')
    print(f'removed_rows={removed_rows.shape}')
    print(f'added_rows={added_rows.shape}')

def test_mutation():

    domain = Domain(['A', 'B', 'C'], [10, 10, 1])
    mutate = get_mutation_fn(domain, muta_rate=3)
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
    mutate2 = get_mutation_fn(domain_large, muta_rate=100)
    mutate2_jit = jax.jit(mutate2)
    x = Dataset.synthetic_jax_rng(domain_large, SYNC_SIZE, rng)
    rng = jax.random.PRNGKey(5)
    for t in range(4):
        rng, rng_sub = jax.random.split(rng)
        stime = time.time()
        x_mutated = mutate2_jit(rng_sub, x)
        x_mutated.block_until_ready()
        print(f'{t}) Elapsed time: {time.time() - stime:.4f}')


from stats import Marginals
# @timeit
def test_jit_ask():
    print(f'test_jit_ask()')
    rounds = 10
    d = 20
    k = 1
    domain = Domain([f'A {i}' for i in range(d)], [3 for _ in range(d)])
    data = Dataset.synthetic(domain, N=10, seed=0)

    domain = data.domain
    print(f'Test jit(ask) with {rounds} rounds. d={d}, k=2')

    marginals = Marginals.get_all_kway_combinations(domain, k=k, bins=[2])
    marginals.fit(data)
    true_stats = marginals.get_true_stats()
    print(f'stat.size = {true_stats.shape}')

    eval_stats_vmap = lambda x: marginals.get_stats_jax_vmap(x)
    eval_stats_vmap = jax.jit(eval_stats_vmap)

    strategy = SimpleGAforSyncDataFast(domain, population_size=200, elite_size=10, data_size=2000,
                                   muta_rate=1,
                                   mate_rate=50, debugging=True)
    stime = time.time()
    key = jax.random.PRNGKey(0)


    state = strategy.initialize(key, eval_stats_vmap)

    # state = strategy.initialize(rng=key)
    print(f'Initialize elapsed time {time.time() - stime:.3f}s')

    print()
    for r in range(rounds):
        stime = time.time()
        # x, a, state = stregy.ask_mate(key, state)
        x, a, state = strategy.ask(key, state)
        # x, a, state = strategy.ask_mate(key, eval_stats_vmap, state)


        x.block_until_ready()

        # fitness = jnp.zeros(200)
        # state = strategy.tell(x, a, fitness, state)
        # if r <= 3 or r == rounds - 1:
        print(f'{r:>3}) Jitted elapsed time {time.time() - stime:.6f}')
        print()


def test_jit_mutate():
    rounds = 10
    d = 3
    k = 1
    domain = Domain([f'A {i}' for i in range(d)], [1 for _ in range(d)])
    data = Dataset.synthetic(domain, N=10, seed=0)
    domain = data.domain
    print(f'Test jit(ask) with {rounds} rounds. d={d}, k=2')
    marginals = Marginals.get_all_kway_combinations(domain, k=k, bins=[2])
    marginals.fit(data)
    # eval_stats_vmap = lambda x: marginals.get_stats_jax_vmap(x)
    get_stats_vmap = jax.jit(marginals.get_stats_jax_vmap)
    strategy = SimpleGAforSyncDataFast(domain, population_size=5, elite_size=2, data_size=10,
                                       muta_rate=1,
                                       mate_rate=1, debugging=True)
    t0 = timer()
    key = jax.random.PRNGKey(0)
    state = strategy.initialize(key, get_stats_vmap)
    t0 = timer(t0, msg=f'init time =')
    archive_stats = strategy.data_size * get_stats_vmap(state.archive)
    archive_error = jnp.abs(state.archive_stats - archive_stats)
    assert archive_error.max() < 0.5, 'something is wrong'

    x, a, state = strategy.ask_mutate(key, state)
    t0 = timer(t0, msg=f'mutate.1 time =')
    strategy.ask_mutate(key, state)
    timer(t0, msg=f'mutate.2 time =')
    x_stats = get_stats_vmap(x)
    diff = a - x_stats * strategy.data_size
    assert jnp.max(jnp.abs(diff)) < 0.5, f'something is wrong. error={jnp.max(jnp.abs(diff)):.5f}'
    print(f'mate test passed!')



    state = strategy.tell_strategy(x, a, fitness=jnp.zeros(5), state=state)
    archive_stats = strategy.data_size * get_stats_vmap(state.archive)
    archive_error = jnp.abs(state.archive_stats - archive_stats)
    assert archive_error.max() < 0.5, 'something is wrong'


def test_jit_mate():
    rounds = 10
    d = 20
    k = 1
    data_size = 2000
    domain = Domain([f'A {i}' for i in range(d)], [3 for _ in range(d)])
    data = Dataset.synthetic(domain, N=10, seed=0)
    domain = data.domain
    print(f'Test jit(ask) with {rounds} rounds. d={d}, k=2')
    marginals = Marginals.get_all_kway_combinations(domain, k=k, bins=[2])
    marginals.fit(data)
    get_stats_vmap = jax.jit(marginals.get_stats_jax_vmap)

    strategy = SimpleGAforSyncDataFast(domain, population_size=200, elite_size=10, data_size=data_size,
                                       muta_rate=1,
                                       mate_rate=123, debugging=True)
    t0 = timer()
    key = jax.random.PRNGKey(0)
    state = strategy.initialize(key, get_stats_vmap)
    t0 = timer(t0, msg=f'init time =')
    archive_stats = strategy.data_size * get_stats_vmap(state.archive)
    archive_error = jnp.abs(state.archive_stats - archive_stats)
    assert archive_error.max() < 0.5, 'something is wrong'
    x, a  = strategy.ask_mate_strategy(key, state)
    t0 = timer(t0, msg=f'mate.1 time =')
    strategy.ask_mate_strategy(key, state)
    timer(t0, msg=f'mate.2 time =')
    x_stats = get_stats_vmap(x)
    diff = a - x_stats * strategy.data_size
    assert jnp.max(jnp.abs(diff)) < 0.5, 'something is wrong'
    print(f'mate test passed!')


if __name__ == "__main__":

    # test_crossover()

    # test_mutation_fn()
    # test_mutation()
    test_jit_ask()
    # test_jit_mutate()
    # test_jit_mate()
