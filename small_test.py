import jax, jax.lax as lax, jax.numpy as jnp, jax.tree_util as tree_util
from jax.experimental.maps import xmap
import time
from backward_ica.stats.hmm import LinearGaussianHMM
from backward_ica.utils import *
from backward_ica.offline_smoothing import check_general_elbo, check_linear_gaussian_elbo
import operator
from functools import reduce
# jax.config.update('jax_disable_jit', True)

enable_x64(True)

weights1 = jnp.load('experiments/online/comparison_naive_and_proposal/weights_method_1.npy')
weights2 = jnp.load('experiments/online/comparison_naive_and_proposal/weights_method_2.npy')

test = 0