#%%
import backward_ica.utils as utils 
import os
from backward_ica.stats.hmm import get_generative_model
from backward_ica.variational import get_variational_model
import dill
import jax, jax.numpy as jnp
import matplotlib.pyplot as plt 

from backward_ica.online_smoothing import OnlinePaRISELBO
from backward_ica.offline_smoothing import GeneralBackwardELBO

experiment_path = 'experiments/p_chaotic_rnn/2023_03_23__23_39_20/neural_backward__online' 
p_and_data_path = os.path.split(experiment_path)[0] 
epoch_nb = 35
num_samples = 10
max_t = 500
utils.enable_x64(True)
jax.config.update('jax_disable_jit', False)

base_key = jax.random.PRNGKey(6)

args_p = utils.load_args('args', p_and_data_path)
args_q = utils.load_args('args', experiment_path)

args_q.state_dim, args_q.obs_dim = args_p.state_dim, args_p.obs_dim

p = get_generative_model(args_p)
q = get_variational_model(args_q)

theta = utils.load_params('theta_star', p_and_data_path)

with open(os.path.join(experiment_path,'data'), 'rb') as f: 
    all_params_through_epochs = dill.load(f)
    # all_params_through_epochs = [d[0] for d in data]


phi = all_params_through_epochs[epoch_nb]

y = jnp.load(os.path.join(p_and_data_path, 'obs_seqs.npy'))[0][:max_t]


offline_elbo = jax.jit(GeneralBackwardELBO(p, q, num_samples).__call__)
online_elbo = jax.jit(OnlinePaRISELBO(p, q, num_samples).batch_compute)

key1, key2 = jax.random.split(base_key, 2)
offline_elbo_value, aux_offline = offline_elbo(key1, y, len(y)-1, p.format_params(theta), q.format_params(phi))
# x_offline = aux_offline[0]
online_elbo_value, aux_online = online_elbo(key2, y, p.format_params(theta), phi)
# x_online = aux_online[-1]

log_weights = aux_online[-1]
max_log_weight_position = jnp.unravel_index(jnp.argmax(log_weights), shape=log_weights.shape)
print(jnp.max(log_weights))
print(max_log_weight_position)
# x_1_offline = x_offline[:,-1,:]
# x_1_online = x_online[-1,:,:]

print(online_elbo_value)
print(offline_elbo_value)

# plt.hist(x_1_offline.flatten())
# plt.show()

# plt.hist(x_1_online.flatten())
# plt.show()
#%%
# print('Offline:', offline_elbo_value)
# print('Online:', online_elbo_value)
# print(online_elbo(key2, y, theta, phi))






