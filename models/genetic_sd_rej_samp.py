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


from models.genetic_sd_new import SimpleGAforSyncData, PopulationState, EvoState

# @dataclass
class GeneticSDRejSampling(Generator):

    def __init__(self,
                 num_generations,
                 domain,
                 population_size,
                 data_size,
                 muta_rate=1,
                 mate_rate=1,
                 print_progress=False,
                 stop_early=True,
                 stop_early_gen=2000,
                 stop_eary_threshold=0,
                 inconsistency_fn=None,
                 inconsistency_weight: float = 1,
                 null_value_frac: float = 0.02,
                 mate_perturbation: float = 0.01
                 ):
        self.domain = domain
        self.data_size = data_size
        self.num_generations = num_generations
        self.print_progress = print_progress
        self.stop_early = stop_early
        self.stop_early_min_generation = stop_early_gen
        self.stop_eary_threshold = stop_eary_threshold

        self.inconsistency_fn = inconsistency_fn
        self.inconsistency_weight = inconsistency_weight
        if self.inconsistency_fn is None:
            self.inconsistency_fn = lambda x: jnp.zeros(x.shape[0])

        self.strategy = SimpleGAforSyncData(domain,
                                            data_size, population_size=population_size,
                                            muta_rate=muta_rate, mate_rate=mate_rate,
                                            null_value_frac=null_value_frac,
                                            mate_perturbation=mate_perturbation)

    def __str__(self):
        return f'GeneticSDV2'

    def fit(self, key, adaptive_statistic: ChainedStatistics,
            sync_dataset: Dataset = None, tolerance: float = 0.0, adaptive_epoch=1):
        """
        Minimize error between real_stats and sync_stats
        """
        W = self.inconsistency_weight
        init_time = time.time()

        selected_noised_statistics = adaptive_statistic.get_selected_noised_statistics()
        selected_statistics = adaptive_statistic.get_selected_statistics_without_noise()
        statistics_fn = jax.jit(adaptive_statistic.get_selected_statistics_fn())
        statistics_fn_debug = adaptive_statistic.get_selected_statistics_fn()

        if self.print_progress:
            gau_error = jnp.abs(selected_noised_statistics - selected_statistics)
            print(f'\tGau Error: Max={gau_error.max():<5.5}\t Average={gau_error.mean():<5.5f}')

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

        # Run GSD to remove violations
        for t in range(self.num_generations):
            # Run GSD using violations to calculate fitness.
            # Run until the best data set has not row violations.

            # ASK (1)
            key, ask_subkey = jax.random.split(key, 2)
            population_state = self.strategy.ask(ask_subkey, state)
            population_state.remove_row.block_until_ready()

            # FIT (1)
            inconsistency_counts: chex.Array
            inconsistency_counts = self.inconsistency_fn(population_state.X)
            fitness = (inconsistency_counts ** 2).sum(axis=1)

            # TELL (1)
            state, rep_best, best_id = self.strategy.tell(population_state.X, fitness, state)
            state.archive.block_until_ready()
            best_fitness = state.best_fitness

            if best_fitness == 0:
                # Stop when the number of violations is zero.
                break




        # Compute fitness of initial elite set
        elite_population_fn = jax.jit(jax.vmap(adaptive_statistic.get_selected_statistics_fn(), in_axes=(0,)))
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
        consistency_time = 0
        tell_time = 0
        last_fitness_debug = None
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
        last_best_fitness = 10000








        for t in range(self.num_generations):

            # ASK (2)
            t0 = timer()
            while True:
                key, ask_subkey = jax.random.split(key, 2)
                population_state = self.strategy.ask(ask_subkey, state)
                inconsistencies = (self.inconsistency_fn(population_state.add_row) * self.data_size).sum()
                if inconsistencies == 0: break

            ask_time += timer() - t0


            # FIT (2)
            t0 = timer()
            fitness = fitness_fn_jit(elite_stat, population_state).block_until_ready()
            fit_time += timer() - t0

            # TELL (2)
            t0 = timer()
            state, rep_best, best_id = self.strategy.tell(population_state.X, fitness, state)
            state.archive.block_until_ready()
            best_fitness = state.best_fitness
            self.fitness_record.append(best_fitness)
            tell_time += timer() - t0

            # UPDATE elite_states (2)
            t0 = timer()
            elite_stat = update_elite_stat_jit(elite_stat, population_state, rep_best, best_id).block_until_ready()
            elite_stat_time += timer() - t0

            if best_fitness < self.stop_eary_threshold: break

            if t % 50 == 0:
                # EARLY STOP
                best_fitness_total = min(best_fitness_total, best_fitness)
                stop_early = False
                if self.stop_early and t > int(self.data_size):
                    if self.early_stop(t, best_fitness_total):
                        stop_early = True

                # DEBUG
                if self.print_progress:
                    if last_fitness_debug is None or best_fitness_total < last_fitness_debug * 0.95 or t >= self.num_generations - 100 or stop_early:
                    # if True:
                        elapsed_time = timer() - init_time
                        X_sync = state.best_member
                        print(f'\tGen {t:05}, fit={best_fitness_total:.6f}, ', end=' ')
                        print(f'\tW={W:.4f}, ', end=' ')
                        t_inf, t_avg, p_l2 = true_loss(X_sync)
                        print(f'\ttrue error(max/avg/l2)=({t_inf:.5f}/{t_avg:.7f}/{p_l2:.3f})', end='')
                        best_inconsistency_count = float((inconsistency_counts[best_id, :] * self.data_size).sum())
                        print(f'\tinconsistencies={best_inconsistency_count:.0f}, ', end=' ')
                        print(f'\t|time={elapsed_time:.4f}(s):', end='')
                        print(f'\task={ask_time:<3.3f}(s), fit={fit_time:<3.3f}(s), tell={tell_time:<3.3f}, ', end='')
                        print(f'elite_stat={elite_stat_time:<3.3f}(s)\t', end='')
                        print(f'consistency={consistency_time:<3.3f}(s)\t', end='')
                        print()
                        last_fitness_debug = best_fitness_total

                if stop_early:
                    if self.print_progress:
                        print(f'\t\tStop early at t={t}')
                    break

        X_sync = state.best_member
        sync_dataset = Dataset.from_numpy_to_dataset(self.domain, X_sync)
        return sync_dataset


