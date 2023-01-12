"""
To run parallelization on multiple cores set
XLA_FLAGS=--xla_force_host_platform_device_count=4
"""
import jax.numpy as jnp
from models import Generator
import time
from stats import PrivateMarginalsState
from typing import Callable
import jax
import chex
from flax import struct
from utils import Dataset, Domain
from functools import partial
from typing import Tuple
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
    def __init__(self, domain: Domain, population_size: int, elite_size: int, data_size: int, muta_rate: int,
                 mate_rate: int,
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
        assert self.muta_rate == 1, "Only supports mutations=1"
        self.mate_rate = mate_rate
        self.debugging = debugging

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
            best_member=init_x[0].astype(jnp.float32),
        )

        rng1, rng2 = jax.random.split(rng, 2)
        random_numbers = jax.random.permutation(rng1, self.data_size, independent=True)
        mute_mate = get_mutate_mating_fn(mate_rate=self.mate_rate, random_numbers=random_numbers)
        self.mate_mutate_vmap = jax.jit(jax.vmap(mute_mate, in_axes=(0, 0, 0, 0)))

        random_numbers2 = jax.random.permutation(rng2, self.data_size, independent=True)
        muta_only = get_mutate_mating_fn(mate_rate=0, random_numbers=random_numbers2)
        self.muta_only_vmap = jax.jit(jax.vmap(muta_only, in_axes=(0, 0, 0, 0)))

        return state

    @partial(jax.jit, static_argnums=(0,))
    def initialize_population(self, rng: chex.PRNGKey):
        d = len(self.domain.attrs)
        pop = Dataset.synthetic_jax_rng(self.domain, self.elite_size * self.data_size, rng)
        initialization = pop.reshape((self.elite_size, self.data_size, d))
        return initialization

    def ask(self, rng: chex.PRNGKey, state: EvoState, mutate_only=False):
        rng_pop, rng_ask = jax.random.split(rng, 2)
        random_data = self.initialize_random_population(rng_pop)
        if mutate_only:
            return self.ask_strategy_mutate_only(rng_ask, random_data, state)
        else:
            return self.ask_strategy(rng_ask, random_data, state)

    @partial(jax.jit, static_argnums=(0,))
    def initialize_random_population(self, rng: chex.PRNGKey):
        d = len(self.domain.attrs)
        pop = Dataset.synthetic_jax_rng(self.domain, self.population_size, rng)
        # initialization = pop.reshape((self.elite_size, self.data_size, d))
        return pop

    @partial(jax.jit, static_argnums=(0,))
    def ask_strategy(self, rng: chex.PRNGKey, random_data, state: EvoState):
        pop_size = self.population_size
        rng, rng_i, rng_j, rng_k, rng_mate, rng_mutate = jax.random.split(rng, 6)
        i = jax.random.randint(rng_i, minval=0, maxval=self.elite_size, shape=(pop_size,))
        j = jax.random.randint(rng_j, minval=0, maxval=self.elite_size, shape=(pop_size,))
        x_i = state.archive[i]
        x_j = state.archive[j]
        a = state.archive_stats[i]

        rng_mate_split = jax.random.split(rng_mate, pop_size)
        x, removed_rows, added_rows = self.mate_mutate_vmap(rng_mate_split, x_i, x_j, random_data)
        return x, a, removed_rows, added_rows, state

    @partial(jax.jit, static_argnums=(0,))
    def ask_strategy_mutate_only(self, rng: chex.PRNGKey, random_data, state: EvoState):
        pop_size = self.population_size
        rng, rng_i, rng_j, rng_k, rng_mate, rng_mutate = jax.random.split(rng, 6)
        i = jax.random.randint(rng_i, minval=0, maxval=self.elite_size, shape=(pop_size,))
        j = jax.random.randint(rng_j, minval=0, maxval=self.elite_size, shape=(pop_size,))
        x_i = state.archive[i]
        x_j = state.archive[j]
        a = state.archive_stats[i]
        rng_muta_split = jax.random.split(rng_mutate, pop_size)
        x, removed_rows, added_rows = self.muta_only_vmap(rng_muta_split, x_i, x_j, random_data)
        return x, a, removed_rows, added_rows, state

    @partial(jax.jit, static_argnums=(0,))
    def tell(
            self,
            x: chex.Array,
            a: chex.Array,
            fitness: chex.Array,
            state: EvoState,
    ) -> EvoState:
        """`tell` performance data for strategy state update."""
        state = self.tell_strategy(x, a, fitness, state)
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


def get_mutate_mating_fn(mate_rate: int, random_numbers):
    muta_rate = 1

    def mute_and_mate(
            rng: chex.PRNGKey, X1: chex.Array, elite_rows: chex.Array, initialization
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        X1 = X1.astype(jnp.float32)
        elite_rows = elite_rows.astype(jnp.float32)

        n, d = X1.shape
        rng, rng1, rng2, rng3, rng4, rng5, rng6 = jax.random.split(rng, 7)

        temp = jax.random.randint(rng1, minval=0, maxval=random_numbers.shape[0] - mate_rate - muta_rate, shape=(1,))
        temp_id = jnp.arange(mate_rate + muta_rate) + temp
        temp_id = jax.random.permutation(rng2, random_numbers[temp_id], independent=True)

        # temp_id = jax.random.choice(rng2, n, replace=False, shape=(mate_rate + muta_rate, ))
        remove_rows_idx = temp_id[:mate_rate]
        mut_rows = temp_id[mate_rate:mate_rate + muta_rate]

        removed_rows_mate = X1[remove_rows_idx, :].reshape((mate_rate, d))
        add_rows_idx = jax.random.randint(rng3, minval=0, maxval=elite_rows.shape[0], shape=(mate_rate,))
        X = X1.at[remove_rows_idx].set(elite_rows[add_rows_idx, :])
        added_rows_mate = elite_rows[add_rows_idx, :].reshape((mate_rate, d))

        ###################
        ## Mutate
        mut_col = jax.random.randint(rng5, minval=0, maxval=d, shape=(muta_rate,))
        values = initialization[mut_col]

        removed_rows_muta = X[mut_rows, :].reshape((muta_rate, d))
        X_mut = X.at[mut_rows, mut_col].set(values)
        added_rows_muta = X_mut[mut_rows, :].reshape((muta_rate, d))

        removed_rows = jnp.concatenate((removed_rows_mate, removed_rows_muta)).reshape((mate_rate + muta_rate, d))
        added_rows = jnp.concatenate((added_rows_mate, added_rows_muta)).reshape((mate_rate + muta_rate, d))

        return X_mut, removed_rows, added_rows

    return mute_and_mate


######################################################################
######################################################################
######################################################################
######################################################################


# @dataclass
class PrivGAfast(Generator):

    def __init__(self,
                 num_generations,
                 strategy: SimpleGAforSyncDataFast,
                 print_progress=False,
                 ):
        self.domain = strategy.domain
        self.data_size = strategy.data_size
        self.num_generations = num_generations
        self.print_progress = print_progress
        self.strategy = strategy

    def __str__(self):
        return f'PrivGAfast'

    def fit(self, key, priv_stat_module: PrivateMarginalsState, init_X=None, tolerance: float = 0.0):
        """
        Minimize error between real_stats and sync_stats
        """

        init_time = time.time()
        # key = jax.random.PRNGKey(seed)
        num_devices = jax.device_count()
        if num_devices > 1:
            print(f'************ {num_devices}  devices found. Using parallelization. ************')

        get_stats_jax = lambda X: priv_stat_module.get_stats(X)
        get_stats_jax_vmap = jax.vmap(get_stats_jax, in_axes=(0,))
        get_stats_jax_vmap = jax.jit(get_stats_jax_vmap)

        self.key, subkey = jax.random.split(key, 2)
        state = self.strategy.initialize(subkey, get_stats_jax_vmap)


        # FITNESS
        priv_stats = priv_stat_module.get_priv_stats()
        get_stats_for_fitness = lambda X: priv_stat_module.get_stats(X)

        def fitness_fn(init_sync_stat, removed_rows, added_rows):
            num_rows, d = removed_rows.shape
            rem_stats = get_stats_for_fitness(removed_rows) * num_rows
            add_stats = get_stats_for_fitness(added_rows) * num_rows
            upt_sync_stat = init_sync_stat + add_stats - rem_stats
            fitness = jnp.linalg.norm(priv_stats - upt_sync_stat / self.data_size, ord=2)
            # fitness = jnp.abs(priv_stats - upt_sync_stat/self.data_size).max()
            return fitness, upt_sync_stat

        # fitness_fn = lambda sync_stat: jnp.linalg.norm(priv_stats - sync_stat, ord=2)
        fitness_vmap_fn = jax.vmap(fitness_fn, in_axes=(0, 0, 0))
        fitness_vmap_fn = jax.jit(fitness_vmap_fn)

        if init_X is not None:
            temp = init_X.reshape((1, init_X.shape[0], init_X.shape[1]))
            temp_stats = self.data_size * get_stats_jax_vmap(temp)
            new_archive = jnp.concatenate([temp, state.archive[1:, :, :]])
            new_archive_stats = jnp.concatenate([temp_stats, state.archive_stats[1:, :]])
            state = state.replace(archive=new_archive, archive_stats=new_archive_stats)

        self.early_stop_init()  # Initiate time-based early stop system

        best_fitness_total = 100000
        ask_time = 0
        fit_time = 0
        tell_time = 0
        last_fitness = None

        mutate_only = 0

        for t in range(self.num_generations):
            self.key, ask_subkey = jax.random.split(self.key, 2)

            # ASK
            t0 = timer()
            x, a, removed_rows, added_rows, state = self.strategy.ask(ask_subkey, state, mutate_only=mutate_only > 0)
            ask_time += timer() - t0

            # FIT
            t0 = timer()
            fitness, a_updated = fitness_vmap_fn(a, removed_rows, added_rows)
            # assert jnp.abs(get_stats_jax_vmap(state.archive) * self.strategy.data_size - state.archive_stats).max() < 1, f'archive stats error'
            fit_time += time.time() - t0

            # TELL
            t0 = timer()
            state = self.strategy.tell(x, a_updated, fitness, state)
            tell_time += timer() - t0

            best_fitness = fitness.min()

            # EARLY STOP
            best_fitness_total = min(best_fitness_total, best_fitness)

            stop_early = False
            if mutate_only == 0 and self.early_stop(best_fitness_total):
                # If early stop is hit for the first type then set a flag to begin mutate only
                if self.print_progress:
                    print(f'\t\tSwitching to mutate only at t={t}')
                mutate_only = mutate_only + 1
            elif mutate_only == 1:
                # if this is the first round where mutate_only is turned on then do nothing
                # This allows time for the jax.jit function to compile
                self.early_stop(best_fitness_total)  # Call this function to reset the time
                mutate_only = mutate_only + 1
            elif mutate_only == 2 and self.early_stop(best_fitness_total):
                # if early stop is hit a second time then halt
                if self.print_progress:
                    print(f'\t\tStop early at t={t}')
                stop_early = True

            if last_fitness is None or best_fitness < last_fitness * 0.99 or t > self.num_generations - 2 or stop_early:
                if self.print_progress:
                    X_sync = state.best_member
                    print(f'\tGeneration {t:05}, best_l2_fitness = {best_fitness:.6f}, ', end=' ')
                    print(
                        f'\t\tprivate (max/l2) error={priv_stat_module.priv_loss_inf(X_sync):.5f}/{priv_stat_module.priv_loss_l2(X_sync):.7f}',
                        end='')
                    print(
                        f'\t\ttrue (max/l2) error={priv_stat_module.true_loss_inf(X_sync):.5f}/{priv_stat_module.true_loss_l2(X_sync):.7f}',
                        end='')
                    print(f'\ttime={timer() - init_time:.4f}(s):', end='')
                    print(f'\task_t={ask_time:.3f}(s), fit_t={fit_time:.3f}(s), tell_t={tell_time:.3f}', end='')
                    print()
                last_fitness = best_fitness

            if stop_early:
                break

        X_sync = state.best_member
        sync_dataset = Dataset.from_numpy_to_dataset(self.domain, X_sync)
        if self.print_progress:
            print(
                f'\t\tFinal private max_error={priv_stat_module.priv_loss_inf(X_sync):.3f}, private l2_error={priv_stat_module.priv_loss_l2(X_sync):.6f},',
                end='\n')

        return sync_dataset


######################################################################
######################################################################
######################################################################
######################################################################
######################################################################

def test_jit_ask():
    from stats import Marginals
    rounds = 5
    d = 20
    k = 1
    print(f'Test jit(ask) with {rounds} rounds. d={d}, k={k}')
    domain = Domain([f'A {i}' for i in range(d)], [3 for _ in range(d)])
    data = Dataset.synthetic(domain, N=10, seed=0)
    domain = data.domain
    marginals = Marginals.get_all_kway_combinations(domain, k=k, bins=[2])
    marginals.fit(data)
    get_stats_vmap = lambda x: marginals.get_stats_jax_vmap(x)
    strategy = SimpleGAforSyncDataFast(domain, population_size=200, elite_size=10, data_size=2000,
                                       muta_rate=1,
                                       mate_rate=0, debugging=True)
    stime = time.time()
    key = jax.random.PRNGKey(0)

    state = strategy.initialize(key, get_stats_vmap)

    # state = strategy.initialize(rng=key)
    print(f'Initialize elapsed time {time.time() - stime:.3f}s')

    for r in range(rounds):
        stime = time.time()
        x, a, removed_rows, added_rows, state = strategy.ask(key, state, mutate_only=True)
        x.block_until_ready()
        a.block_until_ready()
        print(f'{r:>3}) ask() elapsed time {time.time() - stime:.6f}')
        print()

        _, num_rows, d = removed_rows.shape
        rem_stats = get_stats_vmap(removed_rows) * num_rows
        add_stats = get_stats_vmap(added_rows) * num_rows
        a_updated = a + add_stats - rem_stats

        stats = get_stats_vmap(x) * strategy.data_size
        error = jnp.abs(stats - a_updated)
        assert error.max() < 1, f'stats error is {error.max():.1f}'


if __name__ == "__main__":
    # test_crossover()

    # test_mutation_fn()
    # test_mutation()
    test_jit_ask()
    # test_jit_mutate()
