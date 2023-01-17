"""
To run parallelization on multiple cores set
XLA_FLAGS=--xla_force_host_platform_device_count=4
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from models import Generator
import time
from stats import  AdaptiveStatisticState
from typing import Callable
import jax
import chex
from flax import struct
from utils import Dataset, Domain
from functools import partial
from typing import Tuple
from evosax.utils import get_best_fitness_member
from stats import Marginals


def timer(last_time=None, msg=None):
    now = time.time()
    if msg is not None and last_time is not None:
        print(f'{msg} {now - last_time:.5f}')
    return now


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
        # assert self.muta_rate == 1, "Only supports mutations=1"
        self.mate_rate = mate_rate
        self.debugging = debugging

    def initialize(
            self, rng: chex.PRNGKey
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        init_x = self.initialize_elite_population(rng)
        # init_a = self.data_size * get_stats_vmap(init_x)
        state = EvoState(
            mean=init_x.mean(axis=0),
            archive=init_x,
            fitness=jnp.zeros(self.elite_size) + jnp.finfo(jnp.float32).max,
            best_member=init_x[0].astype(jnp.float32),
        )

        rng1, rng2 = jax.random.split(rng, 2)
        random_numbers = jax.random.permutation(rng1, self.data_size, independent=True)
        mute_mate = get_mutate_mating_fn(mate_rate=self.mate_rate, muta_rate=self.muta_rate, random_numbers=random_numbers)
        self.mate_mutate_vmap = jax.jit(jax.vmap(mute_mate, in_axes=(0, 0, 0, 0)))

        random_numbers2 = jax.random.permutation(rng2, self.data_size, independent=True)
        muta_only = get_mutate_mating_fn(mate_rate=0, muta_rate=1, random_numbers=random_numbers2)
        self.muta_only_vmap = jax.jit(jax.vmap(muta_only, in_axes=(0, 0, 0, 0)))

        return state

    @partial(jax.jit, static_argnums=(0,))
    def initialize_elite_population(self, rng: chex.PRNGKey):
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
        pop = Dataset.synthetic_jax_rng(self.domain, self.population_size, rng)
        return pop

    @partial(jax.jit, static_argnums=(0,))
    def ask_strategy(self, rng: chex.PRNGKey, random_data, state: EvoState):
        pop_size = self.population_size
        rng, rng_i, rng_j, rng_k, rng_mate, rng_mutate = jax.random.split(rng, 6)
        # i = jax.random.randint(rng_i, minval=0, maxval=self.elite_size, shape=(pop_size,))
        i = jnp.zeros(shape=(pop_size,)).astype(jnp.int32)
        j = jax.random.randint(rng_j, minval=0, maxval=self.elite_size, shape=(pop_size,))
        x_i = state.archive[i]
        x_j = state.archive[j]
        # a = state.archive_stats[i]

        rng_mate_split = jax.random.split(rng_mate, pop_size)
        x, removed_rows, added_rows = self.mate_mutate_vmap(rng_mate_split, x_i, x_j, random_data)
        return x, i, removed_rows, added_rows, state

    @partial(jax.jit, static_argnums=(0,))
    def ask_strategy_mutate_only(self, rng: chex.PRNGKey, random_data, state: EvoState):
        pop_size = self.population_size
        rng, rng_i, rng_j, rng_k, rng_mate, rng_mutate = jax.random.split(rng, 6)
        i = jax.random.randint(rng_i, minval=0, maxval=self.elite_size, shape=(pop_size,))
        j = jax.random.randint(rng_j, minval=0, maxval=self.elite_size, shape=(pop_size,))
        x_i = state.archive[i]
        x_j = state.archive[j]
        # a = state.archive_stats[i]
        rng_muta_split = jax.random.split(rng_mutate, pop_size)
        x, removed_rows, added_rows = self.muta_only_vmap(rng_muta_split, x_i, x_j, random_data)
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


def get_mutate_mating_fn(mate_rate: int, muta_rate: int, random_numbers):
    # muta_rate = 1

    def mute_and_mate(
            rng: chex.PRNGKey, X1: chex.Array, elite_rows: chex.Array, initialization
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        X1 = X1.astype(jnp.float32)
        elite_rows = elite_rows.astype(jnp.float32)

        n, d = X1.shape
        rng, rng1, rng2, rng3, rng4, rng5, rng6, rng_temp = jax.random.split(rng, 8)

        temp = jax.random.randint(rng1, minval=0, maxval=random_numbers.shape[0] - mate_rate - muta_rate, shape=(1,))
        temp_id = jnp.arange(mate_rate + muta_rate) + temp
        temp_id = jax.random.permutation(rng2, random_numbers[temp_id], independent=True)

        # temp_id = jax.random.choice(rng2, n, replace=False, shape=(mate_rate + muta_rate, ))
        remove_rows_idx = temp_id[:mate_rate]
        mut_rows = temp_id[mate_rate:mate_rate + muta_rate]

        removed_rows_mate = X1[remove_rows_idx, :].reshape((mate_rate, d))
        add_rows_idx = jax.random.randint(rng3, minval=0, maxval=elite_rows.shape[0], shape=(mate_rate,))

        new_rows = elite_rows[add_rows_idx]

        temp = jax.random.randint(rng_temp, minval=0, maxval=2, shape=new_rows.shape)
        # temp = jnp.ones(shape=new_rows.shape)

        added_rows_mate = temp * new_rows + (1-temp) * removed_rows_mate

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


# from stats.halfspaces import
def get_regularization_stat_fn(domain: Domain, regu_get_fn) -> Tuple[chex.Array, Callable, Callable, Callable]:
    # workload_one_way = [(col,) for col in domain.attrs]
    # marginals = Marginals(domain, workload_one_way, bins=[16])
    random_dataset = Dataset.synthetic(domain, N=300, seed=0)
    # marginals.fit(random_dataset)

    regu_fn = regu_get_fn()
    regu_stats = regu_fn(random_dataset.to_numpy())

    elite_reg_fn = jax.jit(jax.vmap(regu_get_fn(), in_axes=(0,)))
    pop_reg1_fn = jax.jit(jax.vmap(regu_get_fn(), in_axes=(0,)))
    pop_reg2_fn = jax.jit(jax.vmap(regu_get_fn(), in_axes=(0,)))

    return regu_stats, elite_reg_fn, pop_reg1_fn, pop_reg2_fn


# @dataclass
class PrivGAfast(Generator):

    def __init__(self,
                 num_generations,
                 strategy: SimpleGAforSyncDataFast,
                 print_progress=False,
                 regu_get_fn=None
                 ):
        self.domain = strategy.domain
        self.data_size = strategy.data_size
        self.num_generations = num_generations
        self.print_progress = print_progress
        self.strategy = strategy

        # regu_fn = jax.jit(stats_module.get_stats_jax)
        # rand_data = Dataset.synthetic(data.domain, N=1000, seed=0)
        # regu_stats = regu_fn(rand_data.to_numpy())
        self.reg_stats, self.elite_reg_fn, self.pop_regu1_fn, self.pop_regu2_fn = get_regularization_stat_fn(self.domain, regu_get_fn)

        self.CACHE = {}

    def __str__(self):
        return f'PrivGAfast'

    def fit(self, key, adaptive_statistic: AdaptiveStatisticState, init_sync=None, tolerance: float = 0.0):
        """
        Minimize error between real_stats and sync_stats
        """

        init_time = time.time()
        # key = jax.random.PRNGKey(seed)
        num_devices = jax.device_count()
        if num_devices > 1:
            print(f'************ {num_devices}  devices found. Using parallelization. ************')

        epoch = len(adaptive_statistic.statistics_ids)
        def get_total_max_error(X):
            sync_stats = adaptive_statistic.STAT_MODULE.get_stats_jax_jit(X)
            max_error = jnp.abs(adaptive_statistic.STAT_MODULE.get_true_stats() - sync_stats).max()
            return max_error

        # elite_fn_list, mate_and_mute_fn_list, mute_onl = []
        for stat_id in adaptive_statistic.statistics_ids:
            if stat_id not in self.CACHE:
                elite_population_fn = adaptive_statistic.STAT_MODULE.get_marginals_fn[stat_id]()
                mate_and_mute_rows_fn = adaptive_statistic.STAT_MODULE.get_marginals_fn[stat_id]()
                mute_only_rows_fn = adaptive_statistic.STAT_MODULE.get_marginals_fn[stat_id]()
                self.CACHE[stat_id] = ((jax.vmap(elite_population_fn, in_axes=(0, ))),
                                       jax.jit(jax.vmap(mate_and_mute_rows_fn, in_axes=(0, ))),
                                       jax.jit(jax.vmap(mute_only_rows_fn, in_axes=(0, )))
                                       )
            # elite_population_fn_jit, mate_and_mute_rows_fn_jit, mute_only_rows_fn_jit = self.CACHE[stat_id]

        debug_fn = lambda X : adaptive_statistic.private_statistics_fn(X)
        debug_fn_vmap = jax.vmap(debug_fn, in_axes=(0, ))

        def elite_population_fn(x):
            pop_stats = []
            for stat_id in adaptive_statistic.statistics_ids:
                elite_population_fn_jit, _, _ = self.CACHE[stat_id]
                pop_stats.append(elite_population_fn_jit(x))
            pop_stats_concat = jnp.concatenate(pop_stats, axis=1)
            return pop_stats_concat

        def mate_and_muta_population_fn(x):
            pop_stats = []
            for stat_id in adaptive_statistic.statistics_ids:
                _, mate_and_mute_rows_fn_jit, _ = self.CACHE[stat_id]
                pop_stats.append(mate_and_mute_rows_fn_jit(x))
            pop_stats_concat = jnp.concatenate(pop_stats, axis=1)
            return pop_stats_concat

        def muta_only_population_fn(x):
            pop_stats = []
            for stat_id in adaptive_statistic.statistics_ids:
                _, _, muta_only_rows_fn_jit = self.CACHE[stat_id]
                pop_stats.append(muta_only_rows_fn_jit(x))
            pop_stats_concat = jnp.concatenate(pop_stats, axis=1)
            return pop_stats_concat


        def regularization_fn(elite_reg_stats, elite_ids, rem_stats, add_stats):
            _, num_queries = elite_reg_stats.shape
            init_sync_stat = elite_reg_stats[elite_ids]
            upt_reg_stat = init_sync_stat + add_stats - rem_stats
            regularization = jnp.linalg.norm(self.reg_stats - upt_reg_stat/self.data_size, ord=2) / (num_queries * self.data_size)
            return regularization, upt_reg_stat
        regularization_vmap_fn = jax.vmap(regularization_fn, in_axes=(None, 0, 0, 0))
        regularization_vmap_fn = jax.jit(regularization_vmap_fn)

        # FITNESS
        priv_stats = adaptive_statistic.get_private_statistics()

        def fitness_fn(elite_stats, elite_ids, rem_stats, add_stats):
            init_sync_stat = elite_stats[elite_ids]
            upt_sync_stat = init_sync_stat + add_stats - rem_stats
            fitness = jnp.linalg.norm(priv_stats - upt_sync_stat/ self.data_size, ord=2)
            return fitness, upt_sync_stat
        fitness_vmap_fn = jax.vmap(fitness_fn, in_axes=(None, 0, 0, 0))
        fitness_vmap_fn = jax.jit(fitness_vmap_fn)



        key, subkey = jax.random.split(key, 2)
        state = self.strategy.initialize(subkey)

        if init_sync is not None:
            temp = init_sync.reshape((1, init_sync.shape[0], init_sync.shape[1]))
            new_archive = jnp.concatenate([temp, state.archive[1:, :, :]])
            state = state.replace(archive=new_archive)
        # assert jnp.abs(elite_population_fn(state.archive) * self.strategy.data_size - state.archive_stats).max() < 1, f'archive stats error'

        # Init slite statistics here
        elite_stats = self.data_size * elite_population_fn(state.archive)
        elite_regu_stats = self.data_size * self.elite_reg_fn(state.archive)

        @jax.jit
        def tell_elite_stats(a, old_elite_stats, new_elite_idx):
            temp_stats = jnp.concatenate([a, old_elite_stats])
            elite_stats = temp_stats[new_elite_idx]
            return elite_stats

        @jax.jit
        def tell_elite_regu_stats(a, old_elite_stats, new_elite_idx):
            temp_stats = jnp.concatenate([a, old_elite_stats])
            elite_stats = temp_stats[new_elite_idx]
            return elite_stats

        self.early_stop_init()  # Initiate time-based early stop system

        best_fitness_total = 100000
        ask_time = 0
        fit_time = 0
        tell_time = 0
        last_fitness = None

        mutate_only = 0

        # last_generation_update = 0
        # last_fitness_update
        total_errors_list = []
        true_train_errors_list = []
        priv_train_errors_list = []
        for t in range(self.num_generations):

            key, ask_subkey = jax.random.split(key, 2)

            # ASK
            t0 = timer()
            x, elite_ids, removed_rows, added_rows, state = self.strategy.ask(ask_subkey, state,
                                                                              mutate_only=mutate_only > 0)
            _, num_rows, _ = removed_rows.shape
            ask_time += timer() - t0

            # FIT
            t0 = timer()
            if num_rows > 1:
                removed_stats = num_rows * mate_and_muta_population_fn(removed_rows)
                added_stats = num_rows * mate_and_muta_population_fn(added_rows)

                removed_reg_stats = num_rows * self.pop_regu1_fn(removed_rows)
                added_reg_stats = num_rows * self.pop_regu1_fn(added_rows)
            else:
                removed_stats = num_rows * muta_only_population_fn(removed_rows)
                added_stats = num_rows * muta_only_population_fn(added_rows)

                removed_reg_stats = num_rows * self.pop_regu2_fn(removed_rows)
                added_reg_stats = num_rows * self.pop_regu2_fn(added_rows)

            fitness, a = fitness_vmap_fn(elite_stats, elite_ids, removed_stats, added_stats)

            regularization_score, a_regu = regularization_vmap_fn(
                elite_regu_stats, elite_ids, removed_reg_stats, added_reg_stats)
            fitness = fitness + regularization_score
            # fitness = fitness


            fit_time += timer() - t0

            # TELL
            t0 = timer()
            state, new_elite_idx = self.strategy.tell(x, fitness, state)
            elite_stats = tell_elite_stats(a, elite_stats, new_elite_idx)
            elite_regu_stats = tell_elite_regu_stats(a_regu, elite_regu_stats, new_elite_idx)

            tell_time += timer() - t0

            best_pop_idx = fitness.argmin()
            best_fitness = fitness[best_pop_idx]
            best_regu_score = regularization_score[best_pop_idx]

            # EARLY STOP
            best_fitness_total = min(best_fitness_total, best_fitness)


            if t > self.data_size:
                if self.early_stop(t, best_fitness_total):
                    if self.print_progress:
                        if mutate_only == 0: print(f'\t\tSwitching to mutate only at t={t}')
                        elif mutate_only == 1: print(f'\t\tStop early at t={t}')
                    mutate_only += 2
            stop_early = mutate_only >= 2


            X_sync = state.best_member
            if t % 10 == 0:
                total_errors_list.append(get_total_max_error(X_sync))
                priv_train_errors_list.append(adaptive_statistic.private_loss_inf(X_sync))
                true_train_errors_list.append(adaptive_statistic.true_loss_inf(X_sync))
                # print(f'\t\t\tg={t:<7}, fitness={best_fitness_total:.5f}, total={total_errors_list[-1]:.5f}, priv_train={priv_train_errors_list[-1]:.5f}, '
                #     f'true_train_errors_list={true_train_errors_list[-1]:.5f},\tmutate_only={mutate_only>0}')
            if last_fitness is None or best_fitness_total < last_fitness * 0.95 or t > self.num_generations - 2 or stop_early:
                if self.print_progress:
                    X_sync = state.best_member
                    gau_error = jnp.abs(adaptive_statistic.get_true_statistics() - adaptive_statistic.get_private_statistics())
                    gau_max = gau_error.max()
                    gau_avg = jnp.linalg.norm(gau_error, ord=2) / gau_error.shape[0]
                    print(f'\tGen {t:05}, fitness={best_fitness_total:.6f}, ', end=' ')
                    print(f'\tregu score={best_regu_score:.6f}, ', end=' ')
                    print(f'\tprivate error(max/l2)=({adaptive_statistic.private_loss_inf(X_sync):.5f}/{adaptive_statistic.private_loss_l2(X_sync):.7f})',end='')
                    print(f'\ttrue error=({adaptive_statistic.true_loss_inf(X_sync):.5f}/{adaptive_statistic.true_loss_l2(X_sync):.7f})',end='')
                    print(f'\tgau error=({gau_max:.5f}/{gau_avg:.7f})',end='')
                    print(f'\t|time={timer() - init_time:.4f}(s):', end='')
                    print(f'\task_t={ask_time:.3f}(s), fit_t={fit_time:.3f}(s), tell_t={tell_time:.3f}', end='')
                    print()
                last_fitness = best_fitness_total

            if stop_early:
                break

        iters = list(np.arange(len(total_errors_list)))
        plt.title(f'epoch={epoch}, mate_rate={self.strategy.mate_rate}')
        plt.plot(iters, total_errors_list, linewidth=3, label='total max error')
        plt.plot(iters, true_train_errors_list, alpha=0.7, linewidth=3, label='true train')
        plt.plot(iters, priv_train_errors_list, alpha=0.7, linewidth=3, label='priv train')
        plt.legend()
        plt.show()

        X_sync = state.best_member
        sync_dataset = Dataset.from_numpy_to_dataset(self.domain, X_sync)
        if self.print_progress:
            print(
                f'\t\tFinal private max_error={adaptive_statistic.private_loss_inf(X_sync):.3f}, private l2_error={adaptive_statistic.private_loss_l2(X_sync):.6f},',
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
    marginals, _ = Marginals.get_all_kway_combinations(domain, k=k, bins=[2])
    marginals.fit(data)
    get_stats_vmap = lambda x: marginals.get_stats_jax_vmap(x)
    strategy = SimpleGAforSyncDataFast(domain, population_size=200, elite_size=10, data_size=2000,
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
            x, a_idx, removed_rows, added_rows, state = strategy.ask(key, state, mutate_only=mutate_only)
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
