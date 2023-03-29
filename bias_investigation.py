#%%
import backward_ica.utils as utils 
import os
from backward_ica.stats.hmm import get_generative_model
from backward_ica.variational import get_variational_model
import dill
import jax, jax.numpy as jnp
import matplotlib.pyplot as plt 
from jax.flatten_util import ravel_pytree

from backward_ica.online_smoothing import OnlineELBO, OnlineELBOAndGrad
from backward_ica.offline_smoothing import GeneralBackwardELBO

experiment_path = 'experiments/p_chaotic_rnn/2023_03_29__18_20_11/neural_backward__online' 
p_and_data_path = os.path.split(experiment_path)[0]
epoch_nb = 1380
num_samples = 100000

utils.enable_x64(True)
jax.config.update('jax_disable_jit', False)

base_key = jax.random.PRNGKey(1)

args_p = utils.load_args('args', p_and_data_path)
args_q = utils.load_args('args', experiment_path)

args_q.state_dim, args_q.obs_dim = args_p.state_dim, args_p.obs_dim

p = get_generative_model(args_p)
q = get_variational_model(args_q)

theta = utils.load_params('theta_star', p_and_data_path)

with open(os.path.join(experiment_path,'data'), 'rb') as f: 
    all_params_through_epochs = dill.load(f)


phi = all_params_through_epochs[epoch_nb]



y = jnp.load(os.path.join(p_and_data_path, 'obs_seqs.npy'))[0]


offline_elbo = GeneralBackwardELBO(p, q, num_samples).__call__
online_elbo = OnlineELBO(p, q, num_samples).batch_compute

key1, key2 = jax.random.split(base_key, 2)


def online(key):
    online_stats, aux = online_elbo(key, y, p.format_params(theta), phi)
    return online_stats['tau'], aux

def offline(key):
    values, aux = offline_elbo(key, y, len(y) - 1, p.format_params(theta), q.format_params(phi))
    return values, aux

@jax.jit
def error_online_and_offline(key):
    online_elbo, online_samples = online(key)
    offline_elbo, offline_aux = offline(key)
    return (online_elbo, online_samples[0]), (offline_elbo, offline_aux[0][:,0,:])

(online_elbo_values, online_samples), (offline_elbo_values, offline_samples) = error_online_and_offline(base_key)
# offlines_samples = jnp.transpose(offline_samples, (1,0,2))
print(online_elbo_values)

print('----')

for dim in range(5):


    plt.hist(online_samples[:,dim], label=f'Online dim {dim}', alpha=0.5)
    plt.hist(offline_samples[:,dim], label=f'Offline dim {dim}', alpha=0.5)
    plt.legend()
    plt.savefig(f'{dim}')
    plt.close()






#%%


# online_elbo_and_grad = OnlineELBOAndGrad(p,q, num_samples).batch_compute


# # offline_elbo_value, aux_offline = offline_elbo(
# #                                         key1, 
# #                                         y, 
# #                                         len(y)-1, 
# #                                         p.format_params(theta), 
# #                                         q.format_params(phi))

# @jax.jit
# def elbo_and_autodiff_grad(key, phi):
#     f = lambda phi: online_elbo(key, y, p.format_params(theta), phi)[0]['tau']
#     elbo, grad = jax.value_and_grad(f)(phi)
#     return elbo, ravel_pytree(grad)[0]

# @jax.jit
# def elbo_and_recursive_grad(key, phi):
#     final_values = online_elbo_and_grad(key, y, p.format_params(theta), phi)[0]

#     return final_values['Omega'], ravel_pytree(final_values['jac_Omega'])[0]

# # @jax.jit
# def compare_elbo_and_autodiff_grad(key): 
        
#     elbo_0, grad_0 = elbo_and_autodiff_grad(key, phi)

#     elbo_1, grad_1 = elbo_and_recursive_grad(key, phi)

#     return elbo_0 - elbo_1, jnp.abs(grad_0 - grad_1).sum() 


# elbo_diffs, grad_diffs = compare_elbo_and_autodiff_grad(key2)
# print(elbo_diffs)
# print('--')
# print(grad_diffs)


# plt.hist(x_1_offline.flatten())
# plt.show()

# plt.hist(x_1_online.flatten())
# plt.show()
#%%
# print('Offline:', offline_elbo_value)
# print('Online:', online_elbo_value)
# print(online_elbo(key2, y, theta, phi))






