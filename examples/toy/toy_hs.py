import itertools
import os

import jax
import jax.numpy as jnp
from models import Generator, PrivGAfast, SimpleGAforSyncDataFast, RelaxedProjectionPP
from stats import Halfspace
from stats.halfspaces_v4 import Halfspace4, Marginals
from toy_datasets.circles import get_circles_dataset
from toy_datasets.moons import get_moons_dataset
from toy_datasets.sparse import get_sparse_dataset
import matplotlib.pyplot as plt
import time
from utils.plot_low_dim_data import plot_2d_data


def run_toy_example(algo,
                    data,
                    stats_module,
                    evaluate_stat_module,
                    epsilon=1.00,
                    seed=0,
                    random_hs=1000,
                    rounds=30,
                    adaptive=False):

    stats_module.fit(data)
    evaluate_stat_module.fit(data)

    folder = f'images/{str(algo)}/{stats_module}/{data_name}/'
    print(f'Saving in {folder}')
    os.makedirs(folder, exist_ok=True)
    def debug_fn(t, tempdata):
        X = tempdata.to_numpy()
        train_error = jnp.abs(stats_module.get_true_stats() - stats_module.get_stats_jax_jit(X)).max()
        eval_error = jnp.abs(evaluate_stat_module.get_true_stats() - evaluate_stat_module.get_stats_jax_jit(X)).max()
        if t == 0:
            save_path = f'{folder}/{data_name}_original.png'
        else:
            save_path = f'{folder}/img_{t:04}.png'

        plot_2d_data(tempdata.to_numpy(), title=f'epoch={t}:\nTrain error = {train_error:.3f}, Eval error={eval_error:.3f}', alpha=0.9, save_path=save_path)





    print(f'Starting {algo}:')
    stime = time.time()
    key = jax.random.PRNGKey(seed)

    if adaptive:
        sync_data = algo.fit_dp_adaptive(key, stat_module=stats_module,  epsilon=epsilon, delta=1e-6,
                                        rounds=rounds, print_progress=True, debug_fn=debug_fn, num_sample=10)
    else:
        sync_data = algo.fit_dp(key, stat_module=stats_module,  epsilon=epsilon, delta=1e-6)

    debug_fn(1, sync_data)
    errors = jax.numpy.abs(stats_module.get_true_stats() - stats_module.get_stats_jit(sync_data))
    ave_error = jax.numpy.linalg.norm(errors, ord=1)/errors.shape[0]
    print(f'{str(algo)}: Train max error = {errors.max():.4f}, ave_error={ave_error:.6f}, time={time.time()-stime:.4f}')


    eval_errors = jax.numpy.abs(evaluate_stat_module.get_true_stats() - evaluate_stat_module.get_stats_jit(sync_data))
    eval_ave_error = jax.numpy.linalg.norm(eval_errors, ord=1)/eval_errors.shape[0]
    print(f'{str(algo)}:  Eval max error = {eval_errors.max():.4f}, ave_error={eval_ave_error:.6f}, time={time.time()-stime:.4f}')



DATASETS = {
    # 'circles': get_circles_dataset(DATA_SIZE=10000),
            # 'moons': get_moons_dataset(DATA_SIZE=10000),
    'sparse': get_sparse_dataset(DATA_SIZE=10000),
    # 'ACSreal' :
}

if __name__ == "__main__":


    for data_name in DATASETS:
        data = DATASETS[data_name]


        algo = PrivGAfast(
            num_generations=100000,
            strategy=SimpleGAforSyncDataFast(
                domain=data.domain,
                data_size=500,
                population_size=100,
                elite_size=5,
                muta_rate=1,
                mate_rate=1,
            ),
            print_progress=False)

        # algo = RelaxedProjectionPP(
        #     domain=data.domain,
        #     data_size=500,
        #     learning_rate=(0.0001,),
        #     print_progress=True)

        hs_stats_module, _ = Halfspace4.get_kway_random_halfspaces(data.domain, k=1, rng=jax.random.PRNGKey(0),
                                                                   random_hs=500)
        eval_hs_stats_module, _ = Halfspace4.get_kway_random_halfspaces(data.domain, k=1, rng=jax.random.PRNGKey(1),
                                                                   random_hs=500)
        # ranges_stat_module, _ = Marginals.get_all_kway_mixed_combinations(data.domain, k_disc=1, k_real=2,
        #                                                                   bins=[2, 4, 8, 16, 32, 64])
        run_toy_example(algo, data,
                        stats_module=hs_stats_module,
                        evaluate_stat_module=eval_hs_stats_module,
                        epsilon=1,
                        rounds=30,
                        adaptive=True)