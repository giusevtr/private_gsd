"""
To run parallelization on multiple cores set
XLA_FLAGS=--xla_force_host_platform_device_count=4
"""
import jax
import jax.numpy as jnp
from models.simple_GA_sync_data import SimpleGAforSyncData
from models import Generator
import time
from utils import Dataset, Domain
from stats import Statistic


class PrivGA(Generator):
    def __init__(self, domain: Domain, stat_module: Statistic, data_size, seed,
                 num_generations: int,
                 popsize: int,
                 top_k: int,
                 sigma_scale: float,
                 crossover: bool,
                 clip_max: float,
                 stop_loss_time_window:int,
                 print_progress: bool):

        # Instantiate the search strategy
        super().__init__(domain, stat_module, data_size, seed)
        self.num_generations = num_generations
        self.popsize = popsize
        self.top_k = top_k
        self.sigma_scale = sigma_scale
        self.crossover = crossover
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
            num_devices=num_devices)

        self.es_params = self.strategy.default_params.replace(
            sigma_init=float(self.sigma_scale),
            # sigma_limit=0.01,
            sigma_limit=float(self.sigma_scale),
            sigma_decay=0.999,
            init_min=0.0,
            init_max=1.0,
            clip_max=clip_max,
            clip_min=0.0
        )

        # Initialize statistics
        self.stat_fn = jax.jit(stat_module.get_stats_fn())

    @staticmethod
    def get_generator(num_generations=100, popsize=20, top_k=5, sigma_scale=0.01, crossover=False,
                       clip_max=1, stop_loss_time_window=10,  print_progress=False):
        generator_init = lambda domain, stat_module, data_size, seed: PrivGA(domain, stat_module, data_size, seed,
                                                                         num_generations=num_generations,
                                                                         popsize=popsize,
                                                                         top_k=top_k,
                                                                         sigma_scale=sigma_scale,
                                                                         crossover=crossover,
                                                                         clip_max=clip_max,
                                                                         stop_loss_time_window=stop_loss_time_window,
                                                                         print_progress=print_progress)
        return generator_init

    def __str__(self):
        return f'SimpleGA: (sigma={self.sigma_scale:.2f}, cross={self.crossover})'

    def fit(self, true_stats, init_X=None):
        """
        Minimize error between real_stats and sync_stats
        """
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

        def std_criterion(fitness):
            """Restart strategy if fitness std across population is small."""
            return fitness.std() < 0.005

        self.key, subkey = jax.random.split(self.key, 2)
        state = self.strategy.initialize(subkey, self.es_params)
        if init_X is not None:
            state = state.replace(mean=init_X.reshape(-1))
        # Run ask-eval-tell loop - NOTE: By default minimization!
        # batch_ask = jax.jit(self.strategy.ask)
        batch_ask = self.strategy.ask
        batch_tell = self.strategy.tell

        init_time = time.time()
        last_fitness = None
        best_fitness_avg = 100000
        last_best_fitness_avg = None

        # Jittable logging helper
        # es_logging = ESLog(num_dims=self.data_size * self.data_dim, num_generations=self.num_generations, top_k=self.top_k, maximize=False)
        # log = es_logging.initialize()

        for t in range(self.num_generations):
            stime = time.time()
            self.key, ask_subkey, eval_subkey = jax.random.split(self.key, 3)
            # x, state = batch_ask(ask_subkey, state, self.es_params)
            x, state = self.strategy.ask_strategy(ask_subkey, state, self.es_params)

            # FITNESS
            fitness = fitness_fn(x)

            state = batch_tell(x, fitness, state, self.es_params)

            best_fitness = fitness.min()

            # print(f't={t:<10} best fitness = {best_fitness:<10.3f}, std={fitness.std()}')
            # Early stop
            best_fitness_avg = min(best_fitness_avg, best_fitness)

            # if std_criterion(fitness):
            #     print('Stopping early')
            #     break

            if t % self.stop_loss_time_window == 0 and t > 0:
                # best_fitness_avg = best_fitness_avg / self.stop_loss_time_window
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
                    print(f'\tsigma={state.sigma:.3f}', end='')
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

