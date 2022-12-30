import itertools
import jax
from models import Generator, RelaxedProjection
from stats import Marginals
from toy_datasets.sparse import get_sparse_dataset
import matplotlib.pyplot as plt
import time
from utils import Dataset
PRINT_PROGRESS = True
ROUNDS = 1
# EPSILON = [0.07]
EPSILON = [1]
SEEDS = [0]


if __name__ == "__main__":

    # rng = np.random.default_rng()
    # data_np = np.column_stack((rng.uniform(low=0.20, high=0.21, size=(10000, )),
    #                            rng.uniform(low=0.30, high=0.80, size=(10000, ))))

    BINS = 30
    data = get_sparse_dataset(DATA_SIZE=10000)
    eval_stats_module = Marginals.get_all_kway_combinations(data.domain, 3, bins=BINS)
    eval_stats_module.fit(data)
    numeric_features = data.domain.get_numeric_cols()

    data_disc = data.discretize(num_bins=BINS)
    train_stats_module = Marginals.get_all_kway_combinations(data_disc.domain, 3)
    train_stats_module.fit(data_disc)

    rap = RelaxedProjection(domain=data_disc.domain, data_size=1000, iterations=5000, learning_rate=0.05, print_progress=True)

    def plot_sparse(data_array, alpha=0.5, title='', save_path=None):
        plt.figure(figsize=(5, 5))
        plt.title(title)
        plt.scatter(data_array[:, 0], data_array[:, 1], c=data_array[:, 2].astype(int), alpha=alpha, s=0.1)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
        plt.close()

    RESULTS = []
    for eps, seed in itertools.product(EPSILON, SEEDS):
        algo: Generator

        def debug_fn(t, sync_dataset):
            data_numeric = Dataset.to_numeric(sync_dataset, numeric_features=numeric_features)
            X = data_numeric.to_numpy()
            plot_sparse(X, title=f'RAP, eps={eps:.2f}, epoch={t:03}')

        ##############
        ## Non-Regularized
        ##############
        stime = time.time()

        key = jax.random.PRNGKey(seed)
        sync_data = rap.fit_dp_adaptive(key, stat_module=train_stats_module, rounds=ROUNDS, epsilon=eps, delta=1e-6,
                                            tolerance=0.0,
                                         print_progress=True, debug_fn=debug_fn)

        numeric_data = Dataset.to_numeric(sync_data, numeric_features)
        erros = eval_stats_module.get_sync_data_errors(numeric_data.to_numpy())

        print(f'RAP: max error = {erros.max():.5f}, time={time.time()-stime}')

        df = rap.ADA_DATA
        df['algo'] = 'RAP'
        df['eps'] = eps
        df['seed'] = seed
        RESULTS.append(df)
