import subprocess
import os
import numpy as np
import pandas as pd
import jax.numpy as jnp
from genetic_sd.utils.dataset_jax import Dataset, Domain
from genetic_sd.utils.statistics import _get_mixed_marginal_fn, _get_bin_edges, _get_density_fn, get_quantiles
from snsynth.utils import cdp_rho

class TestStatistics:

    def test_thresholds(self):
        pass




