from src.hmm import LinearGaussianHMM
import jax.random as random 
from src.kalman import smooth as smooth_jax
from src.kalman import smooth_pykalman
from src.misc import actual_model_from_raw_parameters
from jax import config
config.update("jax_enable_x64", True)

key = random.PRNGKey(0)
key, subkey = random.split(key)
p_raw = LinearGaussianHMM.get_random_model(subkey, state_dim=2, obs_dim=2)
p = actual_model_from_raw_parameters(p_raw)
states, observations = LinearGaussianHMM.sample_joint_sequence(key, p, 8)

result_pykalman = smooth_pykalman(observations, p)
result_jax = smooth_jax(observations, p)

test = 0 