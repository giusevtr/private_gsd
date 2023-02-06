"""
To run parallelization on multiple cores set
XLA_FLAGS=--xla_force_host_platform_device_count=4
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from models import Generator
import time
from stats import AdaptiveStatisticState
import jax
import chex
from flax import struct
from utils import Dataset, Domain, timer
from functools import partial
from typing import Tuple
from evosax.utils import get_best_fitness_member
from models import SimpleGAforSyncData
###############################################################


# @dataclass
class PrivGArel(Generator):

    def __init__(self,
                 num_generations,
                 strategy: SimpleGAforSyncData,
                 print_progress=False,
                 ):
        self.domain = strategy.domain
        self.data_size = strategy.data_size
        self.num_generations = num_generations
        self.print_progress = print_progress
        self.strategy = strategy

        self.CACHE = {}

    def __str__(self):
        return f'PrivGA'

    def fit(self, key, adaptive_statistic: AdaptiveStatisticState, sync_dataset: Dataset=None, tolerance: float = 0.0):
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
        def true_loss(X_arg):
            error = jnp.abs(selected_statistics - statistics_fn(X_arg))
            return jnp.abs(error).max(), error.mean()
        def private_loss(X_arg):
            error = jnp.abs(selected_noised_statistics - statistics_fn(X_arg))
            return jnp.abs(error).max(), error.mean()

        offset = 1 / ( self.data_size)
        def true_relative_loss(X_arg):
            sync_stats = statistics_fn(X_arg)
            relative_losses = jnp.where(sync_stats > selected_statistics,
                              (sync_stats+offset) / (selected_statistics+offset),
                              (selected_statistics+offset) / (sync_stats+offset)
                              )

            return jnp.abs(relative_losses).max(), relative_losses.mean()

        elite_population_fn = jax.vmap(adaptive_statistic.get_selected_statistics_fn(), in_axes=(0, ))
        population1_fn = jax.jit(jax.vmap(adaptive_statistic.get_selected_statistics_fn(), in_axes=(0, )))
        population2_fn = jax.jit(jax.vmap(adaptive_statistic.get_selected_statistics_fn(), in_axes=(0, )))



        # FITNESS
        # priv_stats = adaptive_statistic.get_private_statistics()

        selected_noised_statistics = jnp.clip(selected_noised_statistics, 0, 1)
        def fitness_fn(elite_stats, elite_ids, rem_stats, add_stats):
            init_sync_stat = elite_stats[elite_ids]
            upt_sync_stat = (init_sync_stat + add_stats - rem_stats)
            # fitness = jnp.linalg.norm(selected_noised_statistics - upt_sync_stat / self.data_size, ord=2)

            sync_stats = jnp.clip(upt_sync_stat / self.data_size, 0, 1)

            target_stats = selected_statistics
            # target_stats = selected_noised_statistics
            relative_losses = jnp.where(sync_stats > target_stats,
                                        (sync_stats + offset) / (target_stats + offset),
                                        (target_stats + offset) / (sync_stats + offset)
                                        ) ** 2
            fitness = relative_losses.mean()

            fitness_l2 = jnp.linalg.norm(target_stats - sync_stats, ord=2)**2 / target_stats.shape[0]

            # rad1 = jnp.log(jnp.where(selected_noised_statistics > 0.0, sync_stats / selected_noised_statistics, 100000))
            # rad2 = jnp.log(jnp.where(selected_noised_statistics > 0.0, sync_stats / selected_noised_statistics, 100000))
            # fitness = jnp.dot(sync_stats, rad2)
            # fitness = rad2.mean()


            return fitness + fitness_l2, upt_sync_stat

        fitness_vmap_fn = jax.vmap(fitness_fn, in_axes=(None, 0, 0, 0))
        fitness_vmap_fn = jax.jit(fitness_vmap_fn)
        # fitness_vmap_fn = (fitness_vmap_fn)


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
        # assert jnp.abs(elite_population_fn(state.archive) * self.strategy.data_size - elite_stats).max() < 1, f'archive stats error'

        @jax.jit
        def tell_elite_stats(a, old_elite_stats, new_elite_idx):
            temp_stats = jnp.concatenate([a, old_elite_stats])
            elite_stats = temp_stats[new_elite_idx]
            return elite_stats

        self.early_stop_init()  # Initiate time-based early stop system

        best_fitness_total = 1e18
        ask_time = 0
        fit_time = 0
        tell_time = 0
        last_fitness = None
        mutate_only = 0

        self.fitness_record = []
        for t in range(self.num_generations):

            # ASK
            t0 = timer()
            key, ask_subkey = jax.random.split(key, 2)
            x, elite_ids, removed_rows, added_rows, state = self.strategy.ask(ask_subkey, state,
                                                                              mutate_only=mutate_only > 0)
            _, num_rows, _ = removed_rows.shape
            ask_time += timer() - t0

            # FIT
            t0 = timer()
            if num_rows > 1:
                removed_stats = num_rows * population1_fn(removed_rows)
                added_stats = num_rows * population1_fn(added_rows)

            else:
                removed_stats = num_rows * population2_fn(removed_rows)
                added_stats = num_rows * population2_fn(added_rows)

            fitness, a = fitness_vmap_fn(elite_stats, elite_ids, removed_stats, added_stats)

            fit_time += timer() - t0

            # TELL
            t0 = timer()
            state, new_elite_idx = self.strategy.tell(x, fitness, state)
            elite_stats = tell_elite_stats(a, elite_stats, new_elite_idx)
            best_pop_idx = fitness.argmin()
            best_fitness = fitness[best_pop_idx]
            tell_time += timer() - t0
            self.fitness_record.append(best_fitness)

            # print(f't={t:<3}: ')
            # EARLY STOP
            best_fitness_total = min(best_fitness_total, best_fitness)

            # if t > int(0.25 * self.data_size):
            #     if self.early_stop(t, best_fitness_total):
            #         if self.print_progress:
            #             if mutate_only == 0:
            #                 print(f'\t\tSwitching to mutate only at t={t}')
            #             elif mutate_only == 1:
            #                 print(f'\t\tStop early at t={t}')
            #         mutate_only += 2
            #         if mutate_only>1:
            #             if self.print_progress:
            #                 print(f'\t\tStop early at t={t}')
            # stop_early = mutate_only >= 2

            if last_fitness is None or best_fitness_total < last_fitness * 0.99 or t > self.num_generations - 2 :
                if self.print_progress:
                    elapsed_time = timer() - init_time
                    X_sync = state.best_member
                    print(f'\tGen {t:05}, fitness={best_fitness_total:.6f}, ', end=' ')
                    p_inf, p_avg = private_loss(X_sync)
                    print(f'\tprivate error(max/l2)=({p_inf:.5f}/{p_avg:.7f})',end='')
                    rel_loss_max, rel_loss_avg = true_relative_loss(X_sync)
                    print(f'\ttrue relative error(max/l2)=({rel_loss_max:.2f}/{rel_loss_avg:.3f})',end='')
                    print(f'\t|time={elapsed_time:.4f}(s):', end='')
                    print(f'\task_t={ask_time:.3f}(s), fit_t={fit_time:.3f}(s), tell_t={tell_time:.3f}', end='')
                    print()
                last_fitness = best_fitness_total

            # if stop_early:
            #     break

        X_sync = state.best_member
        # X_sync = state.archive.reshape((-1, X_sync.shape[1]))
        sync_dataset = Dataset.from_numpy_to_dataset(self.domain, X_sync)
        if self.print_progress:
            p_max, p_avg = private_loss(X_sync)
            rel_loss_max, rel_loss_avg = true_relative_loss(X_sync)

            print(f'\t\tFinal private max_error={p_max:.5f}, private l2_error={p_avg:.7f},'
                  f'\ttrue relative error(max/l2)=({rel_loss_max:.2f}/{rel_loss_avg:.3f})',
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
