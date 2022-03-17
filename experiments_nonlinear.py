import backward_ica.hmm as hmm 
from jax import random, numpy as jnp, vmap, tree_util
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

dim_z = p.transition.cov.shape[0]
def rec_net_forward(x):
    net = hk.nets.MLP((32,2*dim_z))
    out = net(x)
    v = out[:dim_z]
    log_W_diag = out[dim_z:]
    return v, -jnp.exp(log_W_diag)

rec_net_init, rec_net_forward = hk.without_apply_rng(hk.transform(rec_net_forward))
key, subkey = random.split(key, 2)
rec_net_params = rec_net_init(rng=key, x=obs_samples[0][0])
rec_net = tree_util.Partial(rec_net_forward, params=rec_net_params)
