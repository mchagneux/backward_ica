import jax 
import jax.numpy as jnp
from jax import config 

config.update("jax_enable_x64", True)

import backward_ica.hmm as hmm
import backward_ica.utils as utils


import argparse 