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
    archive: chex.Array
    fitness: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max


@struct.dataclass
class PopulationState:
    X: chex.Array
    remove_row: chex.Array
    add_row: chex.Array


"""
Implement crossover that is specific to synthetic data
"""


def get_best_fitness_member(
    x: chex.Array, fitness: chex.Array, state
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
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
    return best_member, best_fitness, replace_best, best_in_gen

class SimpleGAforSyncData:
    def __init__(self,
                 domain: Domain,
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
            archive=init_x,
            fitness=jnp.zeros(self.elite_size) + jnp.finfo(jnp.float32).max,
            best_member=init_x[0].astype(jnp.float32),
            best_fitness=jnp.finfo(jnp.float32).max
        )

        rng1, rng2 = jax.random.split(rng, 2)
        random_numbers = jax.random.permutation(rng1, self.data_size, independent=True)
        mute_mate = get_mutate_mating_fn(self.domain, mate_rate=self.mate_rate, muta_rate=self.muta_rate,
                                         random_numbers=random_numbers)
        self.mate_mutate_vmap = jax.jit(jax.vmap(mute_mate, in_axes=(None, 0, 0, 0)))

        return state

    @partial(jax.jit, static_argnums=(0,))
    def initialize_elite_population(self, rng: chex.PRNGKey):
        d = len(self.domain.attrs)
        pop = Dataset.synthetic_jax_rng(self.domain, self.elite_size * self.data_size, rng)
        initialization = pop.reshape((self.elite_size, self.data_size, d))
        return initialization

    @partial(jax.jit, static_argnums=(0,))
    def initialize_random_population(self, rng: chex.PRNGKey):
        pop = Dataset.synthetic_jax_rng(self.domain, self.population_size, rng)
        return pop

    def ask(self, rng: chex.PRNGKey, state: EvoState):
        rng_pop, rng_ask = jax.random.split(rng, 2)
        random_data = self.initialize_random_population(rng_pop)
        return self.ask_strategy(rng_ask, random_data, state)

    @partial(jax.jit, static_argnums=(0,))
    def ask_strategy(self, rng: chex.PRNGKey, random_data, state: EvoState):
        pop_size = self.population_size
        rng, rng_i, rng_j, rng_k, rng_mate, rng_mutate = jax.random.split(rng, 6)
        j = jax.random.randint(rng_j, minval=0, maxval=self.elite_size, shape=(pop_size,))
        x_j = state.archive[j]

        rng_mate_split = jax.random.split(rng_mate, pop_size)
        population = self.mate_mutate_vmap(state.best_member, rng_mate_split, x_j, random_data)
        return population

    @partial(jax.jit, static_argnums=(0,))
    def tell(
            self,
            x: chex.Array,
            fitness: chex.Array,
            state: EvoState,
    ) -> Tuple[EvoState, chex.Array, chex.Array]:
        """`tell` performance data for strategy state update."""
        state = self.tell_strategy(x, fitness, state)
        # return state, is_best_replaced, best_id
        best_member, best_fitness, replace_best, best_in_gen = get_best_fitness_member(x, fitness, state)
        # best_id = jnp.argmin(fitness)
        # best_fitness, best_member = (
        #     fitness[best_id],
        #     x[best_id],
        # )
        return state.replace(
            best_member=best_member,
            best_fitness=best_fitness,
        ), replace_best, best_in_gen

    def tell_strategy(
            self,
            x: chex.Array,
            fitness: chex.Array,
            state: EvoState,
    ) -> EvoState:
        fitness_concat = jnp.concatenate([state.fitness, fitness])
        solution_concat = jnp.concatenate([state.archive, x])
        idx = jnp.argsort(fitness_concat)[0: self.elite_size]

        new_fitness = fitness_concat[idx]
        new_archive = solution_concat[idx]

        new_state = state.replace(
            fitness=new_fitness, archive=new_archive,
        )

        return new_state


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
            X0,
            rng: chex.PRNGKey, elite_rows: chex.Array, initialization
            # ) -> Tuple[chex.Array, chex.Array, chex.Array]:
    ) -> PopulationState:
        X0 = X0.astype(jnp.float32)
        elite_rows = elite_rows.astype(jnp.float32)

        n, d = X0.shape
        rng, rng1, rng2, rng3, rng4, rng5, rng6, rng_temp, rng_normal = jax.random.split(rng, 9)

        temp = jax.random.randint(rng1, minval=0, maxval=random_numbers.shape[0] - mate_rate - muta_rate, shape=(1,))
        temp_id = jnp.arange(mate_rate + muta_rate) + temp
        temp_id = jax.random.permutation(rng2, random_numbers[temp_id], independent=True)

        # temp_id = jax.random.choice(rng2, n, replace=False, shape=(mate_rate + muta_rate, ))
        remove_rows_idx = temp_id[:mate_rate]
        mut_rows = temp_id[mate_rate:mate_rate + muta_rate]

        removed_rows_mate = X0[remove_rows_idx, :].reshape((mate_rate, d))
        add_rows_idx = jax.random.randint(rng3, minval=0, maxval=elite_rows.shape[0], shape=(mate_rate,))

        new_rows = elite_rows[add_rows_idx]
        noise = mask * jax.random.normal(rng_normal, shape=(new_rows.shape[0], d)) * 0.01
        new_rows = new_rows + noise
        new_rows = new_rows.at[:, numeric_idx].set(jnp.clip(new_rows[:, numeric_idx], 0, 1))

        # Only crossover a subset of the values in the rows
        rng_mate1, rng_mate2 = jax.random.split(rng_temp, 2)
        temp = jnp.repeat(jnp.array([1, 0]), jnp.array([1, d - 1]), total_repeat_length=d)
        temp = jnp.repeat(temp.reshape(1, -1), new_rows.shape[0])
        # temp = jax.random.permutation(rng_mate2, temp)
        temp = jax.random.permutation(rng_mate2, temp).reshape((mate_rate, -1))

        added_rows_mate = temp * new_rows + (1 - temp) * removed_rows_mate

        X = X0.at[remove_rows_idx].set(added_rows_mate)

        ###################
        ## Mutate
        mut_col = jax.random.randint(rng5, minval=0, maxval=d, shape=(muta_rate,))
        values = initialization[mut_col]

        removed_rows_muta = X[mut_rows, :].reshape((muta_rate, d))
        X_mut = X.at[mut_rows, mut_col].set(values)
        added_rows_muta = X_mut[mut_rows, :].reshape((muta_rate, d))

        removed_rows = jnp.concatenate((removed_rows_mate, removed_rows_muta)).reshape((mate_rate + muta_rate, d))
        added_rows = jnp.concatenate((added_rows_mate, added_rows_muta)).reshape((mate_rate + muta_rate, d))

        # @struct.dataclass
        # class PopulationState:
        #     X0: chex.Array
        #     X: chex.Array
        #     remove_row: chex.Array
        #     add_row: chex.Array
        pop_state = PopulationState(X=X_mut, remove_row=removed_rows, add_row=added_rows)
        return pop_state

    return mute_and_mate


######################################################################
######################################################################
######################################################################
######################################################################

@struct.dataclass
class TempState:
    state: EvoState
    elite_stat: chex.Array

# @dataclass
class PrivGAJit(Generator):

    def __init__(self,
                 num_generations,
                 domain: Domain,
                 data_size: int = 2000,
                 population_size: int = 100,
                 elite_size: int = 5,
                 muta_rate: int = 1,
                 mate_rate: int = 1,
                 print_progress=False,
                 stop_early=True,

                 stop_eary_threshold=0
                 ):
        self.domain = domain
        self.data_size = data_size
        self.num_generations = num_generations
        self.print_progress = print_progress
        self.strategy = SimpleGAforSyncData(
                                domain,
                                data_size,
                                population_size,
                                elite_size,
                                muta_rate,
                                mate_rate
                            )
        self.stop_early = stop_early

        self.stop_eary_threshold = stop_eary_threshold

    def __str__(self):
        return f'PrivGA'

    def fit(self, key, adaptive_statistic: ChainedStatistics,
            sync_dataset: Dataset = None, tolerance: float = 0.0, adaptive_epoch=1):
        """
        Minimize error between real_stats and sync_stats
        """

        init_time = time.time()

        selected_noised_statistics = adaptive_statistic.get_selected_noised_statistics()
        selected_statistics = adaptive_statistic.get_selected_statistics_without_noise()
        statistics_fn = jax.jit(adaptive_statistic.get_selected_statistics_fn())
        statistics_fn_debug = adaptive_statistic.get_selected_statistics_fn()

        # For debugging
        @jax.jit
        def true_loss(X_arg):
            error = jnp.abs(selected_statistics - statistics_fn_debug(X_arg))
            return jnp.abs(error).max(), jnp.abs(error).mean(), jnp.linalg.norm(error, ord=2)

        @jax.jit
        def private_loss(X_arg):
            error = jnp.abs(selected_noised_statistics - statistics_fn_debug(X_arg))
            return jnp.abs(error).max(), jnp.abs(error).mean(), jnp.linalg.norm(error, ord=2)

        # Create statistic function.
        fitness_statistics_fn = adaptive_statistic.get_selected_statistics_fn()

        def fitness_fn(stats: chex.Array, pop_state: PopulationState):
            # Process one member of the population
            # 1) Update the statistics of this synthetic dataset
            rem_row = pop_state.remove_row
            add_row = pop_state.add_row
            num_rows = rem_row.shape[0]
            add_stats = (num_rows * fitness_statistics_fn(add_row))
            rem_stats = (num_rows * fitness_statistics_fn(rem_row))
            upt_sync_stat = stats.reshape(-1) + add_stats - rem_stats
            # 2) Compute its fitness based on the statistics
            fitness = jnp.linalg.norm(selected_noised_statistics - upt_sync_stat / self.data_size, ord=2) ** 2
            return fitness

        fitness_fn_vmap = jax.vmap(fitness_fn, in_axes=(None, 0))
        fitness_fn_jit = jax.jit(fitness_fn_vmap)

        # INITIALIZE STATE
        key, subkey = jax.random.split(key, 2)

        state = self.strategy.initialize(subkey)

        if sync_dataset is not None:
            init_sync = sync_dataset.to_numpy()
            temp = init_sync.reshape((1, init_sync.shape[0], init_sync.shape[1]))
            new_archive = jnp.concatenate([temp, state.archive[1:, :, :]])
            state = state.replace(archive=new_archive)

        elite_population_fn = jax.vmap(adaptive_statistic.get_selected_statistics_fn(), in_axes=(0,))
        elite_fitness = jnp.linalg.norm(selected_noised_statistics - elite_population_fn(state.archive), axis=1,
                                        ord=2) ** 2
        best_member_id = elite_fitness.argmin()
        state = state.replace(
            fitness=elite_fitness,
            best_member=state.archive[best_member_id],
            best_fitness=elite_fitness[best_member_id]
        )

        self.early_stop_init()  # Initiate time-based early stop system

        best_fitness_total = 100000
        ask_time = 0
        elite_stat_time = 0
        fit_time = 0
        tell_time = 0
        last_fitness = None
        self.fitness_record = []

        if self.print_progress:
            timer(init_time, '\tSetup time = ')

        elite_stat = self.data_size * statistics_fn(state.best_member)  # Statistics of best SD

        update_elite_stat_statistics_fn = adaptive_statistic.get_selected_statistics_fn()
        def update_elite_stat(elite_stat_arg,
                              population_state: PopulationState,
                              replace_best,
                              best_id_arg
                              ):
            num_rows = population_state.remove_row[0].shape[0]

            new_elite_stat = jax.lax.select(
                replace_best,
                elite_stat_arg
                    - (num_rows * update_elite_stat_statistics_fn(population_state.remove_row[best_id_arg]))
                    + (num_rows * update_elite_stat_statistics_fn(population_state.add_row[best_id_arg])),
                elite_stat_arg
            )
            return new_elite_stat

        update_elite_stat_jit = jax.jit(update_elite_stat)

        @jax.jit
        def run_gsd(state_holder_arg: TempState, key_arg: chex.PRNGKey):
            state0 = state_holder_arg.state
            elite_stat0 = state_holder_arg.elite_stat
             # ASK
            key_arg, ask_subkey = jax.random.split(key_arg, 2)
            population_state = self.strategy.ask(ask_subkey, state0)
            # FIT
            fitness0 = fitness_fn_jit(elite_stat0, population_state)
            # TELL
            state0, rep_best, best_id = self.strategy.tell(population_state.X, fitness0, state0)
            # UPDATE elite_states
            elite_stat0 = update_elite_stat_jit(elite_stat0, population_state, rep_best, best_id)

            state_holder_arg = state_holder_arg.replace(state=state0, elite_stat=elite_stat0)
            return state_holder_arg, state0.best_fitness

        def run_gsd_jit(state_holder: TempState, keys):
            return jax.lax.scan(run_gsd, state_holder, keys)

        state_holder = TempState(state=state, elite_stat=elite_stat)

        batch_size = 1000

        for t in range(self.num_generations // batch_size):
            key, keysub = jax.random.split(key, 2)
            state_holder, fitness = run_gsd_jit(state_holder, jax.random.split(keysub, batch_size))

            if fitness[-1] < self.stop_eary_threshold: break



        # state_holder, fitness = run_gsd_jit(state_holder, jax.random.split(key, self.num_generations))

        state = state_holder.state
        X_sync = state.best_member

        elapsed_time = timer() - init_time
        print(f'\t|time={elapsed_time:.4f}(s):', end='\n')
        # p_inf, p_avg, p_l2 = private_loss(X_sync)
        # print(f'\tprivate error(max/avg/l2)=({p_inf:.5f}/{p_avg:.7f}/{p_l2:.3f})', end='')
        t_inf, t_avg, p_l2 = true_loss(X_sync)
        print(f'\ttrue error(max/avg/l2)=({t_inf:.5f}/{t_avg:.7f}/{p_l2:.3f})', end='')
        print()


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
    d = 10
    k = 1
    print(f'Test jit(ask) with {rounds} rounds. d={d}, k={k}')
    domain = Domain([f'A {i}' for i in range(d)], [3 for _ in range(d)])
    data = Dataset.synthetic(domain, N=10, seed=0)
    domain = data.domain
    marginals = Marginals.get_all_kway_combinations(domain, k=k, bins=[2])
    marginal_stat_fn = marginals._get_workload_fn()

    # get_stats_vmap = lambda x: marginals.get_stats_jax_vmap(x)
    get_stats_vmap = jax.vmap(lambda X: marginal_stat_fn(X))

    strategy = SimpleGAforSyncData(domain, population_size=20, elite_size=10, data_size=200,
                                   muta_rate=1,
                                   mate_rate=1, debugging=True)
    t0 = timer()
    key = jax.random.PRNGKey(0)

    strategy.initialize(key)
    t0 = timer(t0, f'Initialize(1) elapsed time')

    state = strategy.initialize(key)
    timer(t0, f'Initialize(2) elapsed time')

    a_archive = get_stats_vmap(state.archive) * strategy.data_size
    for r in range(rounds):
        t0 = timer()
        # x, a_idx, removed_rows, added_rows, state = strategy.ask(key, state)
        population_states, state = strategy.ask(key, state)
        timer(t0, f'{r:>3}) ask() elapsed time')
        print()

        # _, num_rows, d = removed_rows.shape
        # rem_stats = get_stats_vmap(removed_rows) * num_rows
        # add_stats = get_stats_vmap(added_rows) * num_rows
        # # a = a_archive[a_idx]
        # a = a_archive[a_idx] + add_stats - rem_stats
        #
        # stats = get_stats_vmap(x) * strategy.data_size
        # error = jnp.abs(stats - a)
        # assert error.max() < 1, f'stats error is {error.max():.1f}'


if __name__ == "__main__":
    # test_crossover()

    # test_mutation_fn()
    # test_mutation()
    test_jit_ask()
    # test_jit_mutate()