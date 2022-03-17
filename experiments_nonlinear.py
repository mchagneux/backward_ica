import backward_ica.hmm as hmm 
from jax import random, numpy as jnp, vmap, tree_util, value_and_grad, jit
import matplotlib.pyplot as plt 

import optax 
from backward_ica.elbo import get_elbo
import haiku as hk

key = random.PRNGKey(0)
key, *subkeys = random.split(key, 3)
state_dim, obs_dim = 2, 3
num_sequences = 30
sequences_length = 16
#%% Trying the nonlinear case 

p_params, p_def = hmm.get_random_params(subkeys[0], state_dim, obs_dim, 
                                        transition_mapping_type='linear', 
                                        emission_mapping_type='nonlinear')

q_params, q_def = hmm.get_random_params(subkeys[1], state_dim, obs_dim, 
                                        transition_mapping_type='linear', 
                                        emission_mapping_type='linear')

p = hmm.GaussianHMM.build_from_dict(p_params, p_def)

key, *subkeys = random.split(key, num_sequences+1)
state_samples, obs_samples = vmap(p.sample, in_axes=(0, None))(jnp.array(subkeys), sequences_length)


dim_z = p.transition.cov.shape[0]
def rec_net_forward(x):
    net = hk.nets.MLP((32,2*dim_z))
    out = net(x)
    v = out[:dim_z]
    log_W_diag = out[dim_z:]
    return v, -jnp.diag(jnp.exp(log_W_diag))

key, subkey = random.split(key, 2)
rec_net_init, rec_net_apply = hk.without_apply_rng(hk.transform(rec_net_forward))
aux_params = {'rec_net':rec_net_init(subkey, obs_samples[0][0])}
aux_defs = {'rec_net':rec_net_apply}

loss = get_elbo(p_def, q_def, aux_defs)

print(loss(obs_samples[0], p_params, q_params, aux_params))

# optimizer = optax.adam(learning_rate=-1e-3)

# @jit
# def step(p_params, q_params, opt_state, batch):
#     loss_value, grads = value_and_grad(loss, argnums=2)(batch, p_params, q_params)
#     updates, opt_state = optimizer.update(grads, opt_state, q_params)
#     q_params = optax.apply_updates(q_params, updates)
#     return p_params, q_params, opt_state, -loss_value


# def fit(p_params, q_params, aux_params, optimizer: optax.GradientTransformation) -> optax.Params:
#     opt_state = optimizer.init(q_params)

#     eps = jnp.inf
#     old_mean_epoch_elbo = - jnp.mean(vmap(loss, in_axes=(0,None,None))(obs_samples, p_params, q_params))
#     epoch_nb = 0
#     mean_elbos = [old_mean_epoch_elbo - mean_evidence]
#     while eps > 1e-2:
#     # for _ in range(10):
#         epoch_elbo = 0.0
#         for batch in obs_samples: 
#             p_params, q_params, opt_state, elbo_value = step(p_params, q_params, opt_state, batch)
#             epoch_elbo += elbo_value
#         mean_epoch_elbo = epoch_elbo/len(obs_samples)
        
#         eps = jnp.abs(mean_epoch_elbo - old_mean_epoch_elbo)
#         epoch_nb+=1
#         mean_elbos.append(mean_epoch_elbo - mean_evidence)
#         old_mean_epoch_elbo = mean_epoch_elbo
#     return q_params, mean_elbos

# fitted_q_params, mean_elbos = fit(p_params, q_params, optimizer)

# plt.plot(mean_elbos)
# plt.xlabel('Epoch nb'), 
# plt.ylabel('$|\mathcal{L}(\\theta,\\phi)- \log p_\\theta(x)|$')
# plt.show()