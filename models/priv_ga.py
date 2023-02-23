import jax.numpy as jnp
from models import Generator
import time
from stats import ChainedStatistics
import jax
import chex
from flax import struct
from utils import Dataset, Domain, timer
from functools import partial
from typing import Tuple
@struct.dataclass
class EvoState:
    mean: chex.Array
    archive: chex.Array
    # archive_stats: chex.Array
    fitness: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


"""
Implement crossover that is specific to synthetic data
"""

def get_best_fitness_member(
    x: chex.Array, fitness: chex.Array, state
) -> Tuple[chex.Array, chex.Array]:
    best_in_gen = jnp.argmin(fitness)
    best_in_gen_fitness, best_in_gen_member = (
        fitness[best_in_gen],
        x[best_in_gen],
    )
    replace_best = best_in_gen_fitness < state.best_fitness
    best_fitness = jax.lax.select(
        replace_best, best_in_gen_fitness, state.best_fitness
    )
    best_member = jax.lax.select(
        replace_best, best_in_gen_member, state.best_member
    )
    return best_member, best_fitness

class SimpleGAforSyncData:
    def __init__(self, domain: Domain,
                 data_size: int,
                 population_size: int = 100,
                 elite_size: int = 5,
                 muta_rate: int = 1,
                 mate_rate: int = 1,
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
        # assert self.muta_rate == 1, "Only supports mutations=1"
        assert muta_rate > 0, "Mutation rate must be greater than zero."
        assert mate_rate > 0, "Mate rate must be greater than zero."
        self.muta_rate = muta_rate
        self.mate_rate = mate_rate
        self.debugging = debugging

    def initialize(
            self, rng: chex.PRNGKey
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        init_x = self.initialize_elite_population(rng)
        state = EvoState(
            mean=init_x.mean(axis=0),
            archive=init_x,
            fitness=jnp.zeros(self.elite_size) + jnp.finfo(jnp.float32).max,
            best_member=init_x[0].astype(jnp.float32),
        )

        rng1, rng2 = jax.random.split(rng, 2)
        random_numbers = jax.random.permutation(rng1, self.data_size, independent=True)
        mute_mate = get_mutate_mating_fn(self.domain, mate_rate=self.mate_rate, muta_rate=self.muta_rate,
                                         random_numbers=random_numbers)
        self.mate_mutate_vmap = jax.jit(jax.vmap(mute_mate, in_axes=(0, 0, 0, 0)))
        # self.mate_mutate_vmap = (jax.vmap(mute_mate, in_axes=(0, 0, 0, 0)))

        # random_numbers2 = jax.random.permutation(rng2, self.data_size, independent=True)
        # muta_only = get_mutate_mating_fn(self.domain, mate_rate=0, muta_rate=1, random_numbers=random_numbers2)
        # self.muta_only_vmap = jax.jit(jax.vmap(muta_only, in_axes=(0, 0, 0, 0)))

        return state

    @partial(jax.jit, static_argnums=(0,))
    def initialize_elite_population(self, rng: chex.PRNGKey):
        d = len(self.domain.attrs)
        pop = Dataset.synthetic_jax_rng(self.domain, self.elite_size * self.data_size, rng)
        initialization = pop.reshape((self.elite_size, self.data_size, d))
        return initialization

    def ask(self, rng: chex.PRNGKey, state: EvoState):
        rng_pop, rng_ask = jax.random.split(rng, 2)
        random_data = self.initialize_random_population(rng_pop)
        return self.ask_strategy(rng_ask, random_data, state)

    @partial(jax.jit, static_argnums=(0,))
    def initialize_random_population(self, rng: chex.PRNGKey):
        pop = Dataset.synthetic_jax_rng(self.domain, self.population_size, rng)
        return pop

    @partial(jax.jit, static_argnums=(0,))
    def ask_strategy(self, rng: chex.PRNGKey, random_data, state: EvoState):
        pop_size = self.population_size
        rng, rng_i, rng_j, rng_k, rng_mate, rng_mutate = jax.random.split(rng, 6)
        i = jnp.zeros(shape=(pop_size,)).astype(jnp.int32)
        j = jax.random.randint(rng_j, minval=0, maxval=self.elite_size, shape=(pop_size,))
        x_i = state.archive[i]
        x_j = state.archive[j]

        rng_mate_split = jax.random.split(rng_mate, pop_size)
        x, removed_rows, added_rows = self.mate_mutate_vmap(rng_mate_split, x_i, x_j, random_data)
        return x, i, removed_rows, added_rows, state

    @partial(jax.jit, static_argnums=(0,))
    def tell(
            self,
            x: chex.Array,
            # a: chex.Array,
            fitness: chex.Array,
            state: EvoState,
    ) -> Tuple[EvoState, chex.Array]:
        """`tell` performance data for strategy state update."""
        state, new_elite_idx = self.tell_strategy(x, fitness, state)
        best_member, best_fitness = get_best_fitness_member(x, fitness, state)
        return state.replace(
            best_member=best_member,
            best_fitness=best_fitness,
            gen_counter=state.gen_counter + 1,
        ), new_elite_idx

    def tell_strategy(
            self,
            x: chex.Array,
            fitness: chex.Array,
            state: EvoState,
            # elite_popsize: int
    ) -> Tuple[EvoState, chex.Array]:
        fitness = jnp.concatenate([fitness, state.fitness])
        solution = jnp.concatenate([x, state.archive])
        idx = jnp.argsort(fitness)[0: self.elite_size]

        fitness = fitness[idx]
        archive = solution[idx]

        mean = archive.mean(axis=0)
        return state.replace(
            fitness=fitness, archive=archive, mean=mean
        ), idx


def get_mutate_mating_fn(domain: Domain, mate_rate: int, muta_rate: int, random_numbers):
    # muta_rate = 1

    # numeric_idx=jnp.array([0, 1])
    d = len(domain.attrs)
    numeric_idx = domain.get_attribute_indices(domain.get_numeric_cols()).astype(int)
    mask = jnp.zeros(d)
    mask = mask.at[numeric_idx].set(1)
    mask = mask.reshape((1, d))

    cols_ids = jnp.arange(d)

    def mute_and_mate(
            rng: chex.PRNGKey, X1: chex.Array, elite_rows: chex.Array, initialization
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        X1 = X1.astype(jnp.float32)
        elite_rows = elite_rows.astype(jnp.float32)

        n, d = X1.shape
        rng, rng1, rng2, rng3, rng4, rng5, rng6, rng_temp, rng_normal = jax.random.split(rng, 9)

        temp = jax.random.randint(rng1, minval=0, maxval=random_numbers.shape[0] - mate_rate - muta_rate, shape=(1,))
        temp_id = jnp.arange(mate_rate + muta_rate) + temp
        temp_id = jax.random.permutation(rng2, random_numbers[temp_id], independent=True)

        # temp_id = jax.random.choice(rng2, n, replace=False, shape=(mate_rate + muta_rate, ))
        remove_rows_idx = temp_id[:mate_rate]
        mut_rows = temp_id[mate_rate:mate_rate + muta_rate]

        removed_rows_mate = X1[remove_rows_idx, :].reshape((mate_rate, d))
        add_rows_idx = jax.random.randint(rng3, minval=0, maxval=elite_rows.shape[0], shape=(mate_rate,))


        new_rows = elite_rows[add_rows_idx]
        noise = mask * jax.random.normal(rng_normal, shape=(new_rows.shape[0], d)) * 0.01
        new_rows = new_rows + noise
        new_rows = new_rows.at[:, numeric_idx].set(jnp.clip(new_rows[:, numeric_idx], 0, 1))

        # Only crossover a subset of the values in the rows
        rng_mate1, rng_mate2 = jax.random.split(rng_temp, 2)

        # temp = jnp.ones(shape=new_rows.shape)

        # temp = jax.random.randint(rng_temp, minval=0, maxval=2, shape=new_rows.shape)

        # num_row_mates = jax.random.randint(rng_mate1, minval=0, maxval=d, shape=(1, ))
        # temp = jnp.repeat(jnp.array([1, 0]), jnp.array([num_row_mates[0], d - num_row_mates[0]]), total_repeat_length=d)
        num_row_mates = [1]
        temp = jnp.repeat(jnp.array([1, 0]), jnp.array([1, d - 1]), total_repeat_length=d)
        temp = jnp.repeat(temp.reshape(1, -1), new_rows.shape[0])
        temp = jax.random.permutation(rng_mate2, temp)


        # temp = jnp.zeros(shape=new_rows.shape)
        # pos = jax.random.randint(rng_mate1, minval=0, maxval=new_rows.shape[1], shape=(new_rows.shape[0], 2))
        # temp = temp.at[jnp.arange(new_rows.shape[0]), pos].set(1)


        added_rows_mate = temp * new_rows + (1 - temp) * removed_rows_mate

        X = X1.at[remove_rows_idx].set(added_rows_mate)

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
class PrivGA(Generator):

    def __init__(self,
                 num_generations,
                 strategy: SimpleGAforSyncData,
                 print_progress=False,
                 stop_early=True
                 ):
        self.domain = strategy.domain
        self.data_size = strategy.data_size
        self.num_generations = num_generations
        self.print_progress = print_progress
        self.strategy = strategy
        self.stop_early = stop_early

    def __str__(self):
        return f'PrivGA'

    def fit(self, key, adaptive_statistic: ChainedStatistics, sync_dataset: Dataset=None, tolerance: float = 0.0):
        """
        Minimize error between real_stats and sync_stats
        """

        init_time = time.time()
        # key = jax.random.PRNGKey(seed)
        num_devices = jax.device_count()
        if num_devices > 1:
            print(f'************ {num_devices}  devices found. Using parallelization. ************')


        selected_noised_statistics = adaptive_statistic.get_selected_noised_statistics()
        selected_statistics = adaptive_statistic.get_selected_statistics_without_noise()
        statistics_fn = adaptive_statistic.get_selected_statistics_fn()
        # For debugging
        @jax.jit
        def true_loss(X_arg):
            error = jnp.abs(selected_statistics - statistics_fn(X_arg))
            return jnp.abs(error).max(), jnp.abs(error).mean(), jnp.linalg.norm(error, ord=2)

        @jax.jit
        def private_loss(X_arg):
            error = jnp.abs(selected_noised_statistics - statistics_fn(X_arg))
            return jnp.abs(error).max(), jnp.abs(error).mean(), jnp.linalg.norm(error, ord=2)

        elite_population_fn = jax.vmap(adaptive_statistic.get_selected_statistics_fn(), in_axes=(0, ))
        population1_fn = jax.jit(jax.vmap(adaptive_statistic.get_selected_statistics_fn(), in_axes=(0, )))

        def fitness_fn(elite_stats, elite_ids, rem_stats, add_stats):
            init_sync_stat = elite_stats[elite_ids]
            upt_sync_stat = init_sync_stat + add_stats - rem_stats
            fitness = jnp.linalg.norm(selected_noised_statistics - upt_sync_stat / self.data_size, ord=2)**2
            return fitness, upt_sync_stat

        fitness_vmap_fn = jax.vmap(fitness_fn, in_axes=(None, 0, 0, 0))
        fitness_vmap_fn = jax.jit(fitness_vmap_fn)


        if self.print_progress: timer(init_time, '\tSetup time = ')


        t_init = timer()
        key, subkey = jax.random.split(key, 2)
        state = self.strategy.initialize(subkey)

        if sync_dataset is not None:
            init_sync = sync_dataset.to_numpy()
            temp = init_sync.reshape((1, init_sync.shape[0], init_sync.shape[1]))
            new_archive = jnp.concatenate([temp, state.archive[1:, :, :]])
            state = state.replace(archive=new_archive)
        if self.print_progress: timer(t_init, '\tInit strategy time = ')

        # Init slite statistics here

        t_elite = timer()
        elite_stats = self.data_size * elite_population_fn(state.archive)
        if self.print_progress: timer(t_elite, '\tElite population statistics time = ')
        assert jnp.abs(elite_population_fn(state.archive) * self.strategy.data_size - elite_stats).max() < 1, f'archive stats error'

        @jax.jit
        def tell_elite_stats(a, old_elite_stats, new_elite_idx):
            temp_stats = jnp.concatenate([a, old_elite_stats])
            elite_stats = temp_stats[new_elite_idx]
            return elite_stats

        self.early_stop_init()  # Initiate time-based early stop system

        best_fitness_total = 100000
        ask_time = 0
        fit_time = 0
        tell_time = 0
        last_fitness = None

        self.fitness_record = []
        for t in range(self.num_generations):

            # ASK
            t0 = timer()
            key, ask_subkey = jax.random.split(key, 2)
            x, elite_ids, removed_rows, added_rows, state = self.strategy.ask(ask_subkey, state)
            _, num_rows, _ = removed_rows.shape
            ask_time += timer() - t0

            # FIT
            t0 = timer()
            # Compute statistics of rows that changed
            removed_stats = num_rows * population1_fn(removed_rows)
            added_stats = num_rows * population1_fn(added_rows)
            # Fitness of each dataset in the population
            fitness, a = fitness_vmap_fn(elite_stats, elite_ids, removed_stats, added_stats)

            fit_time += timer() - t0

            # TELL
            t0 = timer()
            state, new_elite_idx = self.strategy.tell(x, fitness, state)
            elite_stats = tell_elite_stats(a, elite_stats, new_elite_idx)
            tell_time += timer() - t0

            best_pop_idx = fitness.argmin()
            best_fitness = fitness[best_pop_idx]
            self.fitness_record.append(best_fitness)

            # EARLY STOP
            best_fitness_total = min(best_fitness_total, best_fitness)

            stop_early = False
            if self.stop_early and t > int(0.25 * self.data_size):
                if self.early_stop(t, best_fitness_total):
                    stop_early = True

            if last_fitness is None or best_fitness_total < last_fitness * 0.95 or t > self.num_generations - 2 or stop_early:
                if self.print_progress:
                    elapsed_time = timer() - init_time
                    X_sync = state.best_member

                    print(f'\tGen {t:05}, fitness={best_fitness_total:.6f}, ', end=' ')
                    p_inf, p_avg, p_l2 = private_loss(X_sync)
                    print(f'\tprivate error(max/avg/l2)=({p_inf:.5f}/{p_avg:.7f}/{p_l2:.3f})',end='')
                    t_inf, t_avg, p_l2 = true_loss(X_sync)
                    print(f'\ttrue error(max/avg/l2)=({t_inf:.5f}/{t_avg:.7f}/{p_l2:.3f})',end='')
                    print(f'\t|time={elapsed_time:.4f}(s):', end='')
                    print(f'\task_t={ask_time:.3f}(s), fit_t={fit_time:.3f}(s), tell_t={tell_time:.3f}', end='')
                    print()
                last_fitness = best_fitness_total

            if stop_early:
                if self.print_progress:
                    print(f'\t\tStop early at t={t}')
                break

        X_sync = state.best_member
        sync_dataset = Dataset.from_numpy_to_dataset(self.domain, X_sync)
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
    marginals, _ = Marginals.get_all_kway_combinations(domain, k=k, bins=[2])
    marginals.fit(data)
    get_stats_vmap = lambda x: marginals.get_stats_jax_vmap(x)
    strategy = SimpleGAforSyncData(domain, population_size=200, elite_size=10, data_size=2000,
                                   muta_rate=1,
                                   mate_rate=0, debugging=True)
    t0 = timer()
    key = jax.random.PRNGKey(0)

    strategy.initialize(key)
    t0 = timer(t0, f'Initialize(1) elapsed time')

    state = strategy.initialize(key)
    timer(t0, f'Initialize(2) elapsed time')

    a_archive = get_stats_vmap(state.archive) * strategy.data_size
    for r in range(rounds):
        for mutate_only in [False, True]:
            t0 = timer()
            x, a_idx, removed_rows, added_rows, state = strategy.ask(key, state)
            x.block_until_ready()
            timer(t0, f'{r:>3}) ask(mutate_only={mutate_only}) elapsed time')
            print()

            _, num_rows, d = removed_rows.shape
            rem_stats = get_stats_vmap(removed_rows) * num_rows
            add_stats = get_stats_vmap(added_rows) * num_rows
            # a = a_archive[a_idx]
            a = a_archive[a_idx] + add_stats - rem_stats

            stats = get_stats_vmap(x) * strategy.data_size
            error = jnp.abs(stats - a)
            assert error.max() < 1, f'stats error is {error.max():.1f}'


if __name__ == "__main__":
    # test_crossover()

    # test_mutation_fn()
    # test_mutation()
    test_jit_ask()
    # test_jit_mutate()
