#%%
import backward_ica.utils as utils 
import os
from backward_ica.stats.hmm import get_generative_model
from backward_ica.variational import get_variational_model
import argparse
import jax, jax.numpy as jnp
import matplotlib.pyplot as plt 
from jax.flatten_util import ravel_pytree

from backward_ica.online_smoothing import *
from backward_ica.offline_smoothing import *

experiment_path = 'experiments/p_chaotic_rnn/2023_03_29__19_51_59/neural_backward__offline' 
p_and_data_path = os.path.split(experiment_path)[0]
epoch_nb = 10
num_samples = 1000
T = 2
state_dim, obs_dim = 2,3
utils.enable_x64(True)
jax.config.update('jax_disable_jit', False)

base_key = jax.random.PRNGKey(0)

args_p = argparse.Namespace()
args_q = argparse.Namespace()

args_p.state_dim, args_p.obs_dim = state_dim, obs_dim
args_p.model = 'linear'
args_p.transition_matrix_conditionning = 'diagonal'
args_p.transition_bias = True
args_p.range_transition_map_params = (0.99,1)
args_p = utils.get_defaults(args_p)

args_q.state_dim, args_q.obs_dim = state_dim, obs_dim
args_q.model = 'linear'
args_q.model_options = ''
args_q.transition_matrix_conditionning = 'diagonal'
args_q.transition_bias = True
args_q.range_transition_map_params = (0.7,1)

args_q = utils.get_defaults(args_q)

p = get_generative_model(args_p)
q = get_variational_model(args_q)

check_general_elbo(base_key, p, 1, T, num_samples)
# check_linear_gaussian_elbo(p, 1, T)

key, key_theta, key_phi = jax.random.split(base_key, 3)

theta = p.get_random_params(key_theta)
formatted_theta = p.format_params(theta)

phi = q.get_random_params(key_phi)

key, key_seq = jax.random.split(key, 2)

y = p.sample_seq(key_seq, theta, 500)[1][:T]


oracle_elbo_estimator = LinearGaussianELBO(p, q)
offline_elbo_estimator = GeneralBackwardELBO(p, q, num_samples)
online_elbo_estimator = OnlineELBOAndGrad(p, q, num_samples)

online_elbo_func = lambda key, phi: online_elbo_estimator.batch_compute(
                                                                    key, 
                                                                    y, 
                                                                    formatted_theta, 
                                                                    phi)

offline_elbo_func = lambda key, phi: offline_elbo_estimator(
                                        key, 
                                        y, 
                                        T-1, 
                                        formatted_theta, 
                                        q.format_params(phi))

# @jax.jit
def oracle_elbo_and_grad():
    (oracle_elbo_value, _), oracle_elbo_grad_value = jax.value_and_grad(
                                                    lambda phi: oracle_elbo_estimator(
                                                                    y, 
                                                                    T-1, 
                                                                    formatted_theta,
                                                                    q.format_params(phi)), 
                                                            has_aux=True)(phi)

    return oracle_elbo_value, ravel_pytree(oracle_elbo_grad_value)[0]

# @jax.jit
def offline_elbo_and_grad(key):
    (offline_elbo_value, _), offline_elbo_grad = jax.value_and_grad(
                                                        offline_elbo_func, 
                                                        argnums=1, 
                                                        has_aux=True)(key, phi)
    
    return offline_elbo_value, ravel_pytree(offline_elbo_grad)[0]


# @jax.jit
def online_elbo_and_grad(key):
    online_stats, _ = online_elbo_func(key, phi)
    return online_stats['H'], ravel_pytree(online_stats['F'])[0]



        
oracle_elbo_value, oracle_elbo_grad_value = oracle_elbo_and_grad()
offline_elbo_value, offline_elbo_grad_value = offline_elbo_and_grad(jax.random.PRNGKey(1))
# online_elbo_value, online_elbo_grad_value = online_elbo_and_grad(key)



print('Diff elbo oracle and offline:', oracle_elbo_value - offline_elbo_value)
print('Similarity grad oracle and offline:', utils.cosine_similarity(oracle_elbo_grad_value, offline_elbo_grad_value))

# print('Diff elbo oracle and online:',oracle_elbo_value - online_elbo_value)

# print(utils.cosine_similarity(oracle_elbo_grad_value, online_elbo_grad_value))

# online_estimator = lambda key, phi: ThreePaRIS(
#                             p, 
#                             q, 
#                             utils.state_smoothing_functional(p,q), 
#                             num_samples).batch_compute(key, y, p.format_params(theta), phi)[0]

# # offline_estimator = lambda key: 
# def offline_value_and_grad(key): 
#     return jax.value_and_grad(offline_elbo, argnums=1)(key, phi)

# # offline_estimator(key, y, T-1, p.format_params(theta), q.format_params(phi))[0]


# print(offline_value_and_grad(key))

# def sample_joint(key, y, phi):
#     phi = q.format_params(phi)
#     state_seq = q.compute_state_seq(y, len(y) - 1, phi)
    
#     terminal_params = q.filt_params_from_state(utils.tree_get_idx(-1, state_seq))
#     key, terminal_key = jax.random.split(key, 2)
#     x_T = q.filt_dist.sample(terminal_key, terminal_params)
#     def sample_backwd(x_tp1, key_and_params):
#         key, params = key_and_params
#         x_t = q.backwd_kernel.sample(key, x_tp1, params)
#         return x_t, None

#     return jax.lax.scan(sample_backwd, init=x_T, xs=(utils.tree_get_strides(tree_ge, jax.random.split(key, len(y)-1)), reverse=True)


# x_joint = sample_joint(key, y, phi)
# print(x_joint.shape)
# theta = utils.load_params('theta_star', p_and_data_path)

# with open(os.path.join(experiment_path,'data'), 'rb') as f: 
#     all_params_through_epochs = dill.load(f)

# phi = all_params_through_epochs

# phi = all_params_through_epochs[-1]



# # y = jnp.load(os.path.join(p_and_data_path, 'obs_seqs.npy'))[0][:T]


# # # offline_elbo = GeneralBackwardELBO(p, q, num_samples).__call__
# online_elbo = OnlineELBO(p, q, num_samples).batch_compute
# online_elbo_and_grad = OnlineELBOAndGrad(p,q, num_samples).batch_compute


# #%%




# # # offline_elbo_value, aux_offline = offline_elbo(
# # #                                         key1, 
# # #                                         y, 
# # #                                         len(y)-1, 
# # #                                         p.format_params(theta), 
# # #                                         q.format_params(phi))

# # @jax.jit
# def elbo_and_autodiff_grad(key, phi):
#     f = lambda phi: online_elbo(key, y, p.format_params(theta), phi)[0]['tau']
#     elbo, grad = jax.value_and_grad(f)(phi)
#     return elbo, ravel_pytree(grad)[0]

# # @jax.jit
# def elbo_and_recursive_grad(key, phi):
#     final_values = online_elbo_and_grad(key, y, p.format_params(theta), phi)[0]

#     return final_values['Omega'], ravel_pytree(final_values['jac_Omega'])[0]

# # @jax.jit
# def compare_elbo_and_autodiff_grad(key): 
        
#     elbo_0, grad_0 = elbo_and_autodiff_grad(key, phi)

#     elbo_1, grad_1 = elbo_and_recursive_grad(key, phi)

#     return elbo_0 - elbo_1, utils.cosine_similarity(grad_0, grad_1)


# elbo_diffs, grad_diffs = compare_elbo_and_autodiff_grad(base_key)
# print(elbo_diffs)
# print('--')
# print(grad_diffs)
# # print(grad_ratios)




# # plt.hist(x_1_offline.flatten())
# # plt.show()

# # plt.hist(x_1_online.flatten())
# # plt.show()
#%%
# print('Offline:', offline_elbo_value)
# print('Online:', online_elbo_value)
# print(online_elbo(key2, y, theta, phi))






