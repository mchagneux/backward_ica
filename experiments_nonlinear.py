import backward_ica.hmm as hmm 
from jax import random, numpy as jnp, vmap
import haiku as hk

key = random.PRNGKey(0)
key, subkey = random.split(key, 2)
state_dim, obs_dim = 2, 3
num_sequences = 30
sequences_length = 16
#%% Trying the nonlinear case 

p_params, p_aux = hmm.get_random_params(subkey, state_dim, obs_dim, 
                                        transition_mapping_type='linear', 
                                        emission_mapping_type='nonlinear')

p = hmm.GaussianHMM.build_from_dict(p_params, p_aux)

key, *subkeys = random.split(key, num_sequences+1)
state_samples, obs_samples = vmap(p.sample, in_axes=(0, None))(jnp.array(subkeys), sequences_length)

def rec_net_forward(x):
    net = hk.nets.MLP((64,1))
    return net(x)

rec_net = hk.transform(rec_net_forward)
key, subkey = random.split(key, 2)
rec_net_params = rec_net.init(key, )
