import itertools
import jax
import jax.numpy as jnp
from models import Generator, RelaxedProjection
from stats import Marginals, Halfspace
from dev.toy_datasets.sparse import get_sparse_dataset
import time
from utils import Dataset
from plot import plot_sparse
PRINT_PROGRESS = True
ROUNDS = 1
# EPSILON = [0.07]
EPSILON = [1]
SEEDS = [0]


if __name__ == "__main__":

    # rng = np.random.default_rng()
    # data_np = np.column_stack((rng.uniform(low=0.20, high=0.21, size=(10000, )),
    #                            rng.uniform(low=0.30, high=0.80, size=(10000, ))))

    BINS = 16
    bins = [2, 4, 8, 16, 32]

    learning_rate = 0.3
    data = get_sparse_dataset(DATA_SIZE=10000)


    key_hs = jax.random.PRNGKey(0)
    halfspace_stats_module, _ = Halfspace.get_kway_random_halfspaces(data.domain, k=1, rng=key_hs,
                                                                           random_hs=1000)
    halfspace_stats_module.fit(data)



    eval_stats_module, kway_combinations = Marginals.get_all_kway_mixed_combinations(data.domain, k_disc=1, k_real=2,
                                                                                bins=bins)
    print(kway_combinations)
    # eval_stats_module = Marginals.get_all_kway_combinations(data.domain, 3, bins=[2, 4, 8, 16, 32])
    eval_stats_module.fit(data)
    numeric_features = data.domain.get_numeric_cols()

    data_disc = data.discretize(num_bins=BINS)
    train_stats_module = Marginals(data_disc.domain, kway_combinations)
    train_stats_module.fit(data_disc)

    rap = RelaxedProjection(domain=data_disc.domain, data_size=1000, iterations=5000,
                            learning_rate=learning_rate, print_progress=True)

    plot_sparse(data.to_numpy(), title='Original sparse')


    RESULTS = []
    for eps, seed in itertools.product(EPSILON, SEEDS):
        algo: Generator

        def debug_fn(t, sync_dataset):
            data_numeric = Dataset.to_numeric(sync_dataset, numeric_features=numeric_features)
            X = data_numeric.to_numpy()
            plot_sparse(X, title=f'RAP, lr={learning_rate:.2f} eps={eps:.2f}, epoch={t:03}')

        ##############
        ## Non-Regularized
        ##############
        stime = time.time()

        key = jax.random.PRNGKey(seed)
        sync_data = rap.fit_dp_adaptive(key, stat_module=train_stats_module,
                                        rounds=ROUNDS,
                                        epsilon=eps, delta=1e-6,
                                        tolerance=0.0,
                                        print_progress=True,
                                        debug_fn=debug_fn)

        numeric_data = Dataset.to_numeric(sync_data, numeric_features)
        errors = eval_stats_module.get_sync_data_errors(numeric_data.to_numpy())

        stats = eval_stats_module.get_stats_jit(numeric_data)
        ave_error = jax.numpy.linalg.norm(eval_stats_module.get_true_stats() - stats, ord=1)
        print(f'RAP. Marginals: max error={errors.max():.5f}, ave_error={ave_error:.6f} time={time.time() - stime:.4f}')


        # Halfspace error
        hs_error = jnp.abs(halfspace_stats_module.get_true_stats() - halfspace_stats_module.get_stats_jit(sync_data))
        print(f'RAP: Halfspace: max error={hs_error.max():.5f}, ave_error={jnp.linalg.norm(hs_error):.6f}')
        df = rap.ADA_DATA
        df['algo'] = 'RAP'
        df['eps'] = eps
        df['seed'] = seed
        RESULTS.append(df)
