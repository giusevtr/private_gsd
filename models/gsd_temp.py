import jax.numpy as jnp
import pandas as pd
from models import Generator
import time
from stats import ChainedStatistics
import jax
import chex
from flax import struct
from utils import Dataset, Domain, timer
from functools import partial
from typing import Tuple



from models.gsd import SimpleGAforSyncData, EvoState, PopulationState

# @dataclass
class GSDtemp(Generator):

    def __init__(self,
                 num_generations,
                 domain,
                 data_size,
                 population_size_muta=50,
                 population_size_cross=50,
                 population_size=None,
                 muta_rate=1,
                 mate_rate=1,
                 print_progress=False,
                 stop_early=True,
                 stop_early_gen=None,
                 stop_eary_threshold=0,
                 sparse_statistics=False
                 ):
        self.domain = domain
        self.data_size = data_size
        self.num_generations = num_generations
        self.print_progress = print_progress
        self.stop_early = stop_early
        self.stop_eary_threshold = stop_eary_threshold
        self.sparse_statistics = sparse_statistics
        self.stop_early_min_generation = stop_early_gen if stop_early_gen is not None else data_size
        self.strategy = SimpleGAforSyncData(domain, data_size,
                                            population_size_muta=population_size_muta,
                                            population_size_cross=population_size_cross,
                                            population_size=population_size,
                                            muta_rate=muta_rate, mate_rate=mate_rate)
        self.stop_generation = None

    def __str__(self):
        return f'GSD'

    def fit(self, key, adaptive_statistic: ChainedStatistics,
            sync_dataset: Dataset = None, tolerance: float = 0.0, adaptive_epoch=1):
        """
        Minimize error between real_stats and sync_stats
        """

        self.stop_generation = None

        if self.sparse_statistics:
            selected_statistics, selected_noised_statistics, statistics_fn = adaptive_statistic.get_selected_trimmed_statistics_fn()
            if self.print_progress:
                print(f'Number of sparse statistics is {selected_statistics.shape[0]}. Time = {timer() - init_time:.2f}')
        else:
            selected_noised_statistics = adaptive_statistic.get_selected_noised_statistics()
            statistics_fn = adaptive_statistic.get_selected_statistics_fn()

        return self.fit_function(key, selected_noised_statistics, statistics_fn, sync_dataset)

    def fit_function(self, key, selected_noised_statistics, statistics_fn, sync_dataset: Dataset = None, tolerance: float = 0.0):

        init_time = timer()

        @jax.jit
        def private_loss(X_arg):
            error = jnp.abs(selected_noised_statistics - statistics_fn(X_arg))
            return jnp.abs(error).max(), jnp.abs(error).mean(), jnp.linalg.norm(error, ord=2)

        def fitness_fn(stats: chex.Array, pop_state: PopulationState):
            # Process one member of the population
            # 1) Update the statistics of this synthetic dataset
            rem_row = pop_state.remove_row
            add_row = pop_state.add_row
            num_rows = rem_row.shape[0]
            add_stats = (num_rows * statistics_fn(add_row))
            rem_stats = (num_rows * statistics_fn(rem_row))
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

        elite_population_fn = jax.vmap(statistics_fn, in_axes=(0,))
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

        def update_elite_stat(elite_stat_arg,
                              population_state: PopulationState,
                              replace_best,
                              best_id_arg
                              ):
            num_rows = population_state.remove_row[0].shape[0]

            new_elite_stat = jax.lax.select(
                    replace_best,
                    elite_stat_arg
                        - (num_rows * statistics_fn(population_state.remove_row[best_id_arg]))
                        + (num_rows * statistics_fn(population_state.add_row[best_id_arg])),
                    elite_stat_arg
                )
            return new_elite_stat

        update_elite_stat_jit = jax.jit(update_elite_stat)
        LAST_LAG_FITNESS = state.best_fitness
        true_results = []
        for t in range(self.num_generations):
            self.stop_generation = t  # Update the stop generation
            # ASK
            t0 = timer()
            key, ask_subkey = jax.random.split(key, 2)
            population_state = self.strategy.ask(ask_subkey, state)
            population_state.remove_row.block_until_ready()
            ask_time += timer() - t0

            # FIT
            t0 = timer()
            fitness = fitness_fn_jit(elite_stat, population_state).block_until_ready()
            fit_time += timer() - t0

            # TELL
            t0 = timer()
            state, rep_best, best_id = self.strategy.tell(population_state.X, fitness, state)
            state.archive.block_until_ready()
            best_fitness = state.best_fitness

            self.fitness_record.append([t, best_fitness, timer() - init_time])

            tell_time += timer() - t0
            # UPDATE elite_states
            elite_stat = update_elite_stat_jit(elite_stat, population_state, rep_best, best_id).block_until_ready()
            elite_stat_time += timer() - t0

            if best_fitness < self.stop_eary_threshold: break
            if (t % self.stop_early_min_generation) == 0 and t > self.stop_early_min_generation and self.stop_early:
                loss_change = jnp.abs(LAST_LAG_FITNESS - state.best_fitness) / LAST_LAG_FITNESS

                if loss_change < 0.0001:
                    if self.print_progress: print(f'\t\t ### Stop early at {t} ###')
                    break
                LAST_LAG_FITNESS = state.best_fitness

            if (t % 500 == 0) and self.print_progress:
                # DEBUG
                best_fitness_total = min(best_fitness_total, best_fitness)
                if last_fitness is None or best_fitness_total < last_fitness * 0.99 or t > self.num_generations - 2:
                    elapsed_time = timer() - init_time
                    X_sync = state.best_member
                    print(f'\tGen {t:05}, fit={best_fitness_total:.6f}, ', end=' ')
                    print(f'\t|time={elapsed_time:.4f}(s):', end='')
                    print(f'\task={ask_time:<3.3f}(s), fit={fit_time:<3.3f}(s), tell={tell_time:<3.3f}, ', end='')
                    print(f'elite_stat={elite_stat_time:<3.3f}(s)\t', end='')
                    print()
                    last_fitness = best_fitness_total

        # Save progress for debugging.
        self.true_results_df = pd.DataFrame(true_results, columns=['G', 'Max', 'Avg', 'L2'])
        X_sync = state.best_member
        sync_dataset = Dataset.from_numpy_to_dataset(self.domain, X_sync)
        return sync_dataset

