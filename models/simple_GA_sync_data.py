import jax
import jax.numpy as jnp
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
    sigma: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    cross_over_rate: float = 0.5
    sigma_init: float = 0.07
    sigma_decay: float = 0.999
    sigma_limit: float = 0.01
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max







"""
Implement crossover that is specific to synthetic data
"""
class SimpleGAforSyncData:
    def __init__(self,
                 domain: Domain,
                 data_size: int, # number of synthetic data rows
                 generations:int,
                 popsize: int,
                 # sync_data_shape: tuple,
                 elite_ratio: float = 0.5,
                 crossover=True,
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
        self.num_devices = num_devices
        self.crossover = crossover
        self.domain = domain

        # self.sparsity, self.min_sparsity, self.sparsity_decay = self.perturbation_sparsity

        self.mate_vmap = jax.jit(jax.vmap(single_mate, in_axes=(0, 0, 0)))
    @property
    def default_params(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        return EvoParams()
    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        return EvoParams()

    @partial(jax.jit, static_argnums=(0,))
    def initialize(
        self, rng: chex.PRNGKey, params: Optional[EvoParams] = None
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params

        # Initialize strategy based on strategy-specific initialize method
        state = self.initialize_strategy(rng, params)
        return state

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: EvoParams
    ) -> EvoState:
        """`initialize` the differential evolution strategy."""
        # initialization = jax.random.uniform(
        #     rng,
        #     (self.elite_popsize, self.num_dims),
        #     minval=params.init_min,
        #     maxval=params.init_max,
        # )
        # temp =  [Dataset.synthetic(self.domain, self.data_size, s).to_numpy() for s in range(self.elite_popsize)]

        temp = []
        rng_np = np.random.default_rng(0)
        for s in range(self.elite_popsize):
            data = Dataset.synthetic_rng(self.domain, self.data_size, rng_np)
            temp.append(data.to_numpy())

        initialization = jnp.array(temp)
        state = EvoState(
            mean=initialization.mean(axis=0),
            archive=initialization,
            fitness=jnp.zeros(self.elite_popsize) + jnp.finfo(jnp.float32).max,
            sigma=params.sigma_init,
            best_member=initialization[0],
        )
        return state

    @partial(jax.jit, static_argnums=(0,))
    def ask(
        self,
        rng: chex.PRNGKey,
        state: EvoState,
        params: Optional[EvoParams] = None,
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        x, state = self.ask_strategy(rng, state, params)
        return x, state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """
        `ask` for new proposed candidates to evaluate next.
        1. For each member of elite:
          - Sample two current elite members (a & b)
          - Cross over all dims of a with corresponding one from b
            if random number > co-rate
          - Additionally add noise on top of all elite parameters
        """
        num_elites_to_keep = 1

        rng, rng_eps, rng_idx_a, rng_idx_b, rng_cross = jax.random.split(rng, 5)

        archive_size = state.archive.shape[0]
        num_mates = self.popsize - archive_size
        rng_mate = jax.random.split(rng, num_mates)

        elite_ids = jnp.arange(self.elite_popsize)
        idx_a = jax.random.choice(rng_idx_a, elite_ids, (num_mates,))
        idx_b = jax.random.choice(rng_idx_b, elite_ids, (num_mates,))
        members_a = state.archive[idx_a]
        members_b = state.archive[idx_b]

        if self.crossover:
            # cross_over_rates = jnp.zeros(shape=(num_mates,))
            x = self.mate_vmap(
                    rng_mate, members_a, members_b
                )
        else:
            x = members_a[:num_mates]
            # cross_over_rates = jax.random.uniform(rng_cross, shape=(num_mates,))


        # Add archive
        x = jnp.vstack([x, state.archive])
        x = mutation_strategy(rng_eps, x, state.sigma, domain=self.domain)
        # epsilon = get_perturbation(rng_eps, self.popsize, self.sync_data_shape, state.sigma)
        # x += epsilon

        # ADD best
        x = jnp.vstack([x[num_elites_to_keep:, :, :], state.archive[:num_elites_to_keep, :, :]])
        # return jnp.squeeze(x), state
        return x, state



    @partial(jax.jit, static_argnums=(0,))
    def tell(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: Optional[EvoParams] = None,
    ) -> chex.ArrayTree:
        """`tell` performance data for strategy state update."""

        # Update the search state based on strategy-specific update
        state = self.tell_strategy(x, fitness, state, params)

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
        params: EvoParams,
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
        sigma = jax.lax.select(
            state.sigma > params.sigma_limit,
            state.sigma * params.sigma_decay,
            state.sigma,
        )
        # Keep mean across stored archive around for evaluation protocol
        mean = archive.mean(axis=0)
        return state.replace(
            fitness=fitness, archive=archive, sigma=sigma, mean=mean
        )


def get_perturbation(rng, col, domain: Domain, sigma):
    cat_cols = domain.get_categorical_cols()
    cat_idx = domain.get_attribute_indices(cat_cols)
    rng1, rng2, rng3 = jax.random.split(rng, 3)

    if col in cat_idx:
        return jnp.choose(rng, jnp.arange(domain.size(col)))
    else:
        return jax.random.normal(rng3, (1, )) * sigma


def mutation_strategy(rng: chex.PRNGKey, x, sigma, domain: Domain):
    pop_size, n, d = x.shape

    # cat_cols = domain.get_categorical_cols()
    # cat_idx = domain.get_attribute_indices(cat_cols)
    # x_reshaped = x.reshape((-1, n, d))

    rng1, rng2, rng3 = jax.random.split(rng, 3)

    # For each individual, select a row and a column
    row_mutation = jax.random.randint(rng1, minval=0, maxval=n, shape=(pop_size, 1))
    col_mutation = jax.random.randint(rng2, minval=0, maxval=d, shape=(pop_size, 1))

    cat_cols = domain.get_categorical_cols()

    new_x = []
    # cat_mutations = jnp.zeros(shape=(pop_size, 1))
    ones = jnp.ones(shape=(pop_size, 1))
    i = jnp.array([[i] for i in range(pop_size)])
    for col, att in enumerate(domain.attrs):
        temp = (col_mutation == col).astype(dtype=jnp.int32).squeeze().reshape(-1, 1, 1)
        # cat_mutations = jnp.logical_or(cat_mutations, temp)

        rng3, rng3_sub = jax.random.split(rng3, 2)
        # pop_indices = jnp.argwhere(col_mutation == col)
        # rows = row_mutation[pop_indices]
        if att in cat_cols:
            values = jax.random.randint(rng3_sub, minval=0, maxval=domain.size(att), shape=(pop_size, 1))
        else:
            values = jax.random.uniform(rng3_sub, shape=(pop_size, 1))

        x1 = temp * x.at[i, row_mutation, col_mutation].set(values)
        x2 = (1-temp) * x
        x = x1 + x2

        # new_x.append(x.at[pop_indices, rows, col].set(values))

    # Sample noise
    # pertub_vmap = jax.vmap(get_perturbation, in_axes=(0, 0, None, None))
    # rng_perturb = jax.random.split(rng3, pop_size)
    # new_values = pertub_vmap(rng_perturb, col_mutation, domain, sigma)
    # x_reshaped = x_reshaped.at[i, row_mutation, col_mutation].set(new_values)

    return x

# def get_perturbation(rng: chex.PRNGKey, pop, sync_data_shape, sigma):
#     rng1, rng2, rng3 = jax.random.split(rng, 3)
#     n, d = sync_data_shape
#
#     # For each individual, select a row and a column
#     row_mutation = jax.random.randint(rng1, minval=0, maxval=n, shape=(pop, 1))
#     col_mutation = jax.random.randint(rng2, minval=0, maxval=d, shape=(pop, 1))
#
#     epsilon = (jax.random.normal(rng3, (pop, 1)) * sigma)
#     temp = jnp.zeros((pop, n, d))
#
#     i = jnp.array([[i] for i in range(pop)])
#     epsilon = temp.at[i, row_mutation, col_mutation].add(epsilon)
#     epsilon = epsilon.reshape((pop, n * d))
#     return epsilon


def single_mate(
    rng: chex.PRNGKey, a: chex.Array, b: chex.Array,
) -> chex.Array:
    """Only cross-over dims for x% of all dims."""
    n, d = a.shape
    # n, d = sync_data_shape

    X = a
    Y = b

    # rng1, rng2 = jax.random.split(rng, 2)
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
    cross_over_rate = 0.1 * jax.random.uniform(rng1, shape=(1,))

    idx = (jax.random.uniform(rng2, (n, )) > cross_over_rate).reshape((n, 1))
    X = jax.random.permutation(rng3, X, axis=0)
    Y = jax.random.permutation(rng4, Y, axis=0)

    XY = X * (1 - idx) + Y * idx
    cross_over_candidate = XY
    return cross_over_candidate




    # cross_rate = jax.random.uniform(rng1, (1,))
    # n_cross = jnp.asarray([(n-1) * cross_rate + 1]).astype(int)
    #
    # Y = jax.random.permutation(rng2, Y, axis=0)
    # Y = Y[:n_cross, :]
    #
    # # idx = (jax.random.uniform(rng1, (X.shape[0],)) > cross_over_rate).reshape((X.shape[0], 1))
    # # X = jax.random.permutation(rng2, X, axis=0)
    # # Y = jax.random.permutation(rng3, Y, axis=0)
    #
    # # sa
    # # data_size = len(a)
    # # avg = jnp.concatenate([a, b])
    # # cross_over_candidate = jax.random.choice(rng, avg, shape=(data_size,), replace=False)
    # XY = jnp.vstack((X, Y))
    # XY = jax.random.permutation(rng2, XY, axis=0)
    # XY = XY[:n, :]
    #
    # # XY = X * (1 - idx) + Y * idx
    # cross_over_candidate = XY.reshape(-1)
    # return cross_over_candidate



if __name__ == "__main__":

    key = jax.random.PRNGKey(0)


    domain = Domain(['A', 'B'], [10, 3])
    strategy = SimpleGAforSyncData(domain=domain,
                 data_size=2, # number of synthetic data rows
                 generations=2,
                 popsize=10)
    es_params = strategy.default_params.replace(
        sigma_init=float(0.5),
    )
    state = strategy.initialize_strategy(key, es_params)
    # x = state.archive


    x, state = strategy.ask_strategy(key, state, es_params)

    print(x)

    # print('before')
    # print(x)
    # new_x = mutation_strategy(key, x, 0.5, domain)
    # print('after')
    # print(x)
    # print(f'different coordinates.')
    # x2 = x.reshape(5, -1)
    # new_x2 = new_x.reshape(5, -1)
    # diff = x2 != new_x2
    # print(jnp.abs(diff).sum(axis=1))
    # print('')