"""
To run parallelization on multiple cores set
XLA_FLAGS=--xla_force_host_platform_device_count=4
"""
import jax
import jax.numpy as jnp
from models import Generator
import time
from utils import Dataset, Domain
from stats import Statistic


class PrivGA(Generator):
    def __init__(self, domain: Domain,
                 stat_module: Statistic,
                 data_size,
                 seed,
                 num_generations: int,
                 popsize: int,
                 top_k: int,
                 crossover: int,
                 mutations: int,
                 stop_loss_time_window:int,
                 print_progress: bool):

        # Instantiate the search strategy
        super().__init__(domain, stat_module, data_size, seed)
        self.num_generations = num_generations
        self.popsize = popsize
        self.top_k = top_k
        self.crossover = crossover
        self.mutations = mutations
        self.elite_ratio = top_k / popsize
        self.print_progress = print_progress
        self.stop_loss_time_window = stop_loss_time_window

        num_devices = jax.device_count()

        self.strategy = SimpleGAforSyncData(
            domain=domain,
            data_size=data_size,
            generations=num_generations,
            popsize=self.popsize,
            elite_ratio=self.elite_ratio,
            crossover=crossover,
            mutations=mutations,
            num_devices=num_devices)

        # Initialize statistics
        self.stat_fn = jax.jit(stat_module.get_stats_fn())

        # self.init_population_fn = jax.jit(get_initialize_population)


    @staticmethod
    def get_generator(num_generations=100, popsize=20, top_k=5, crossover: int =1, mutations:int =1,
                       clip_max=1, stop_loss_time_window=10,  print_progress=False):
        generator_init = lambda domain, stat_module, data_size, seed: PrivGA(domain,
                                                                            stat_module,
                                                                            data_size,
                                                                            seed,
                                                                            num_generations=num_generations,
                                                                            popsize=popsize,
                                                                            top_k=top_k,
                                                                            crossover=crossover,
                                                                            mutations=mutations,
                                                                            stop_loss_time_window=stop_loss_time_window,
                                                                            print_progress=print_progress)
        return generator_init

    def __str__(self):
        return f'SimpleGA(cross={self.crossover}, mut={self.mutations})'

    def fit(self, true_stats, init_X=None):
        """
        Minimize error between real_stats and sync_stats
        """
        init_time = time.time()
        num_devices = jax.device_count()
        if num_devices>1:
            print(f'************ {num_devices}  devices found. Using parallelization. ************')

        # FITNESS
        compute_error_fn = lambda X: (jnp.linalg.norm(true_stats - self.stat_fn(X), ord=2)**2 ).squeeze()
        compute_error_vmap = jax.jit(jax.vmap(compute_error_fn, in_axes=(0, )))

        def distributed_error_fn(X):
            return compute_error_vmap(X)
        compute_error_pmap = jax.pmap(distributed_error_fn, in_axes=(0, ))

        def fitness_fn(X):
            """
            Evaluate the error of the synthetic data
            """
            if num_devices == 1:
                return compute_error_vmap(X)
            X_distributed = X.reshape((num_devices, -1, self.data_size, self.data_dim))
            fitness = compute_error_pmap(X_distributed)
            fitness = jnp.concatenate(fitness)
            return fitness.squeeze()

        stime = time.time()
        self.key, subkey = jax.random.split(self.key, 2)
        state = self.strategy.initialize(subkey)
        if init_X is not None:
            state = state.replace(mean=init_X.reshape(-1))
        print(f'population initialization time = {time.time() - stime:.3f}')

        last_fitness = None
        best_fitness_avg = 100000
        last_best_fitness_avg = None

        for t in range(self.num_generations):
            stime = time.time()
            self.key, ask_subkey, eval_subkey = jax.random.split(self.key, 3)
            x, state = self.strategy.ask(ask_subkey, state)

            if t == 0:
                print(f'ask time = {time.time() - stime:.3f}')

            # FITNESS
            fitness = fitness_fn(x)
            if t == 0:
                print(f'fitness time = {time.time() - stime:.3f}')

            state = self.strategy.tell(x, fitness, state)
            best_fitness = fitness.min()
            if t == 0:
                print(f'tell time = {time.time() - stime:.3f}')

            # Early stop
            best_fitness_avg = min(best_fitness_avg, best_fitness)

            if t % self.stop_loss_time_window == 0 and t > 0:
                if last_best_fitness_avg is not None:
                    percent_change = jnp.abs(best_fitness_avg - last_best_fitness_avg) / last_best_fitness_avg
                    if percent_change < 0.001:
                        print('Stop early ast iteration', t)
                        break

                last_best_fitness_avg = best_fitness_avg
                best_fitness_avg = 100000

            if last_fitness is None or best_fitness < last_fitness * 0.95 or t > self.num_generations-2 :
                if self.print_progress:
                    X_sync = state.best_member
                    errors = true_stats - self.stat_fn(X_sync)
                    max_error = jnp.abs(errors).max()

                    print(f'\tGeneration {t}, best_l2_fitness = {jnp.sqrt(best_fitness):.3f}, ', end=' ')
                    print(f'\ttime={time.time() -init_time:.3f}(s):', end='')
                    print(f'\t\tmax_error={max_error:.3f}', end='')
                    # print(f'\tsigma={state.sigma:.3f}', end='')
                    print()


                last_fitness = best_fitness

            # if self.print_progress:
            #     log = es_logging.update(log, x, fitness)
        # # Save best.
        # if self.print_progress:
        #     es_logging.plot(log, "ES", ylims=(0, 30))

        self.key, rng_final = jax.random.split(self.key, 2)
        X_sync = state.best_member

        sync_dataset = Dataset.from_numpy_to_dataset(self.domain, X_sync)
        return sync_dataset

import jax
import numpy as np
import chex
from typing import Tuple
from evosax.strategy import Strategy
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
    def __init__(self,
                 domain: Domain,
                 data_size: int, # number of synthetic data rows
                 generations:int,
                 popsize: int,
                 crossover: int = 1,
                 mutations: int = 1,
                 elite_ratio: float = 0.5,
                 num_devices=1
                 ):
        """Simple Genetic Algorithm For Synthetic Data Search Adapted from (Such et al., 2017)
        Reference: https://arxiv.org/abs/1712.06567
        Inspired by: https://github.com/hardmaru/estool/blob/master/es.py"""

        d = len(domain.attrs)
        num_dims = d * data_size
        self.data_size = data_size
        self.popsize = popsize
        # super().__init__(num_dims, popsize)
        self.domain = domain
        self.generations = generations
        # self.sync_data_shape = sync_data_shape
        self.elite_ratio = elite_ratio
        self.elite_popsize = max(1, int(self.popsize * self.elite_ratio))
        self.strategy_name = "SimpleGA"
        self.crossover = crossover
        self.mutations = mutations
        self.num_devices = num_devices
        self.domain = domain

        # self.sparsity, self.min_sparsity, self.sparsity_decay = self.perturbation_sparsity

        self.mate_vmap = jax.jit(jax.vmap(single_mate, in_axes=(0, 0, 0, None)), static_argnums=3)

        mutate = get_mutation_fn(domain, self.mutations)
        self.mutate_vmap = jax.jit(jax.vmap(mutate, in_axes=(0, 0)))

    @partial(jax.jit, static_argnums=(0,))
    def initialize(
        self, rng: chex.PRNGKey
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        # Initialize strategy based on strategy-specific initialize method
        state = self.initialize_strategy(rng)
        return state

    def initialize_strategy(self, rng: chex.PRNGKey) -> EvoState:
        """`initialize` the differential evolution strategy."""
        initialization = initialize_population(rng, self.elite_popsize, self.domain, self.data_size).astype(jnp.float32)

        state = EvoState(
            mean=initialization.mean(axis=0),
            archive=initialization,
            fitness=jnp.zeros(self.elite_popsize) + jnp.finfo(jnp.float32).max,
            best_member=initialization[0],
        )
        return state

    @partial(jax.jit, static_argnums=(0,))
    def ask(
        self,
        rng: chex.PRNGKey,
        state: EvoState,
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        x, state = self.ask_strategy(rng, state)
        return x, state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState
    ) -> Tuple[chex.Array, EvoState]:
        """
        """

        rng, rng_eps, rng_idx_a, rng_idx_b, rng_cross = jax.random.split(rng, 5)

        archive_size = state.archive.shape[0]
        num_mates = self.popsize - archive_size
        rng_mate = jax.random.split(rng, num_mates)

        elite_ids = jnp.arange(self.elite_popsize)
        idx_a = jax.random.choice(rng_idx_a, elite_ids, (num_mates,))
        idx_b = jax.random.choice(rng_idx_b, elite_ids, (num_mates,))
        members_a = state.archive[idx_a]
        members_b = state.archive[idx_b]

        if self.crossover > 0:
            # cross_over_rates = jnp.zeros(shape=(num_mates,))
            x = self.mate_vmap(
                    rng_mate, members_a, members_b, self.crossover
                )
        else:
            x = members_a[:num_mates]

        # Add archive
        x = jnp.vstack([x, state.archive])

        rng_mutate = jax.random.split(rng, self.popsize)
        x = self.mutate_vmap(rng_mutate, x)

        # ADD best
        # x = jnp.vstack([x[num_elites_to_keep:, :, :], state.archive[:num_elites_to_keep, :, :]])
        # return jnp.squeeze(x), state
        return x.astype(jnp.float32), state


    @partial(jax.jit, static_argnums=(0,))
    def tell(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
    ) -> chex.ArrayTree:
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
    ) -> EvoState:
        """
        `tell` update to ES state.
        If fitness of y <= fitness of x -> replace in population.
        """
        # Combine current elite and recent generation info
        fitness = jnp.concatenate([fitness, state.fitness])
        solution = jnp.concatenate([x, state.archive])
        # Select top elite from total archive info
        idx = jnp.argsort(fitness)[0: self.elite_popsize]

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
    # seed = np.array(jax.random.randint(rng, shape=(1,), minval=0, maxval=1000000))[0]
    temp = []
    # rng_np = np.random.default_rng(seed)
    for s in range(pop_size):
        # data = Dataset.synthetic_rng(domain, data_size, rng_np)
        rng, rng_sub = jax.random.split(rng)
        X = Dataset.synthetic_jax_rng(domain, data_size, rng_sub)
        temp.append(X)

    initialization = jnp.array(temp)
    return initialization


def get_mutation_fn(domain: Domain, mutations: int):

    def mutate(rng: chex.PRNGKey, X):
        n, d = X.shape
        rng1, rng2 = jax.random.split(rng, 2)
        total_params = n * d
        idx = jnp.concatenate((jnp.ones(mutations), jnp.zeros(total_params-mutations)))
        idx = jax.random.permutation(rng1, idx)
        idx = idx.reshape((n, d))
        initialization = Dataset.synthetic_jax_rng(domain, n, rng2)
        X = X * (1-idx) + initialization * idx
        return X

    return mutate


def single_mate(
    rng: chex.PRNGKey, X: chex.Array, Y: chex.Array, crossover
) -> chex.Array:
    """Only cross-over dims for x% of all dims."""
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
    X = jax.random.permutation(rng3, X, axis=0)
    Y = jax.random.permutation(rng4, Y, axis=0)
    n, d = X.shape
    # n, d = sync_data_shape

    idx = jnp.concatenate((jnp.ones(crossover), jnp.zeros(n-crossover))).reshape((n, 1))

    XY = X * (1 - idx) + Y * idx
    cross_over_candidate = XY
    return cross_over_candidate


######################################################################
######################################################################
######################################################################
######################################################################
######################################################################

def test_crossover():

    domain = Domain(['A', 'B'], [10, 3])
    rng = np.random.default_rng(0)
    X = jnp.array(Dataset.synthetic_rng(domain, 5, rng).to_numpy())
    Y = jnp.array(Dataset.synthetic_rng(domain, 5, rng).to_numpy())

    rng_jax = jax.random.PRNGKey(0)
    XY = single_mate(rng_jax, X, Y, crossover=2)

    print(X)
    print(Y)
    print(XY)

def test_mutation():

    domain = Domain(['A', 'B', 'C'], [10, 10, 1])

    mutate = jax.jit(get_mutation_fn(domain, mutations=2))
    rng = jax.random.PRNGKey(2)
    # x = initialize_population(rng, pop_size=3, domain=domain, data_size=4)

    x = Dataset.synthetic_jax_rng(domain, 4, rng)

    rng, rng_sub = jax.random.split(rng)
    x2 = mutate(rng_sub, x)

    print('x =')
    print(x)
    print('x2=')
    print(x2)


if __name__ == "__main__":
    # test_crossover()

    test_mutation()