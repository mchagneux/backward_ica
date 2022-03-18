import backward_ica.hmm as hmm 
from jax import random, numpy as jnp, vmap, tree_util, value_and_grad, jit
import matplotlib.pyplot as plt 
from backward_ica.kalman import filter as kalman_filter

import optax 
from backward_ica.elbo import NonLinearELBO, get_elbo
import haiku as hk

key = random.PRNGKey(0)
key, *subkeys = random.split(key, 3)
state_dim, obs_dim = 2, 3
num_sequences = 16
sequences_length = 30
#%% Trying the nonlinear case 

p_params, p_def = hmm.get_random_params(subkeys[0], state_dim, obs_dim, 
                                        transition_mapping_type='linear', 
                                        emission_mapping_type='linear')

# q_params, q_def = hmm.get_random_params(subkeys[1], state_dim, obs_dim, 
#                                         transition_mapping_type='linear', 
#                                         emission_mapping_type='linear')

p = hmm.GaussianHMM.build_from_dict(p_params, p_def)

key, *subkeys = random.split(key, num_sequences+1)
state_samples, obs_samples = vmap(p.sample, in_axes=(0, None))(jnp.array(subkeys), sequences_length)

evidence_function = lambda sequence, p: kalman_filter(sequence, p)[-1]
evidence_sequences = vmap(evidence_function, in_axes=(0, None))(obs_samples, p)
mean_evidence = jnp.mean(evidence_sequences)


dim_z = p.transition.cov.shape[0]

def rec_net_forward(x):
    net = hk.nets.MLP((64,2*dim_z))
    out = net(x)
    v = out[:dim_z]
    log_W_diag = out[dim_z:]
    return v, jnp.diag(log_W_diag)

key, subkey = random.split(key, 2)
rec_net_init, rec_net_apply = hk.without_apply_rng(hk.transform(rec_net_forward))
aux_params = {'rec_net':rec_net_init(subkey, obs_samples[0][0])}
aux_defs = {'rec_net':rec_net_apply}

loss = lambda obs, params: NonLinearELBO(p_def, p_def, aux_defs).compute(obs, p_params, p_params, params)


# print(loss(obs_samples[0], aux_params))

optimizer = optax.adam(learning_rate=-1e-6)

@jit
def step(params, opt_state, batch):
    loss_value, grads = value_and_grad(loss, argnums=1)(batch, params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value


def fit(params, optimizer: optax.GradientTransformation) -> optax.Params:
    opt_state = optimizer.init(params)

    mean_elbo = jnp.mean(vmap(loss, in_axes=(0,None))(obs_samples, params))
    epoch_nb = 0
    dist_to_evidence = [jnp.abs(mean_elbo-mean_evidence)]
    # while jnp.abs(dist_to_evidence[-1]) > 1e-2:
    for _ in range(10000):
        epoch_elbo = 0.0
        for batch in obs_samples: 
            params, opt_state, elbo_value = step(params, opt_state, batch)
            epoch_elbo += elbo_value
        mean_elbo = epoch_elbo/obs_samples.shape[0]
        # eps = jnp.abs(mean_elbo - old_mean_elbo)
        epoch_nb+=1
        dist_to_evidence.append(jnp.abs(mean_elbo-mean_evidence))
        print(dist_to_evidence[-1])
    return params, dist_to_evidence

# params = aux_params
params, dist_to_evidence = fit(aux_params, optimizer)

plt.plot(dist_to_evidence)
plt.xlabel('Epoch nb'), 
plt.ylabel('$|\mathcal{L}(\\theta,\\phi)- \log p_\\theta(x)|$')
plt.show()