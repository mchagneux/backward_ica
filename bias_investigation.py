#%%
import backward_ica.utils as utils 
import os
from backward_ica.stats.hmm import get_generative_model
from backward_ica.variational import get_variational_model
import argparse
import jax, jax.numpy as jnp
import matplotlib.pyplot as plt 
from jax.flatten_util import ravel_pytree
jax.config.update('jax_platform_name', 'cpu')
from backward_ica.online_smoothing import *
from backward_ica.offline_smoothing import *
import pandas as pd 

experiment_path = 'experiments/p_chaotic_rnn/2023_04_03__15_03_01/johnson_backward' 
p_and_data_path = os.path.split(experiment_path)[0]
num_samples = 100
num_samples_oracle = 100000
T = 1
utils.enable_x64(True)

jax.config.update('jax_disable_jit', False)

base_key = jax.random.PRNGKey(0)
args_p = utils.load_args('args',p_and_data_path)
args_q = utils.load_args('args', experiment_path)
args_q.state_dim, args_q.obs_dim = args_p.state_dim, args_p.obs_dim
num_runs = 100

p = get_generative_model(args_p)
q = get_variational_model(args_q)


theta = utils.load_params('theta_star', p_and_data_path)
with open(os.path.join(experiment_path, 'data'), 'rb') as f: 
    phi = dill.load(f)[-1]

y = jnp.load(os.path.join(p_and_data_path, 
                          'obs_seqs.npy')).squeeze()[:T]

OfflineSmoother = OfflineVariationalAdditiveSmoothing(
                                                p, 
                                                q, 
                                                utils.offline_elbo_functional(p,q), 
                                                num_samples_oracle)



OnlineSmoother = OnlineVariationalAdditiveSmoothing(
                                            p, 
                                            q, 
                                            init_carry_gradients_F, 
                                            init_gradients_F,
                                            update_gradients_F,
                                            utils.online_elbo_functional(p,q),
                                            exp_and_normalize,
                                            num_samples)

OnlineSmootherAutodiff = OnlineVariationalAdditiveSmoothing(
                                                    p, 
                                                    q, 
                                                    init_carry,
                                                    init_PaRIS,
                                                    update_PaRIS,
                                                    utils.online_elbo_functional(p,q),
                                                    exp_and_normalize,
                                                    num_samples)


OnlineSmootherAutodiffFromRecursions = OnlineVariationalAdditiveSmoothing(
                                                    p, 
                                                    q, 
                                                    init_carry_gradients_reparam,
                                                    init_gradients_reparam,
                                                    update_gradients_reparam,
                                                    utils.online_elbo_functional(p,q),
                                                    exp_and_normalize,
                                                    num_samples)



key, subkey = jax.random.split(base_key, 2)

@jax.jit
def offline_smoothing_value_and_grad(key):
    def f(phi):
        return OfflineSmoother(key, y, T-1, p.format_params(theta), q.format_params(phi))[0]
    value, grad = jax.value_and_grad(f)(phi)
    return value, ravel_pytree(grad)[0]

@jax.jit
def online_smoothing_value_and_grad(key):
    
    def f(phi):
        results = OnlineSmoother.batch_compute(key, y, p.format_params(theta), phi)[0]
        H = results['stats']['H']
        F = vmap_ravel(results['stats']['F'])
        grad_log_q = vmap_ravel(results['grad_log_q'])
        elbo = jnp.mean(H, axis=0) / len(y)
        grad = jnp.mean(jax.vmap(lambda a,b,c: a*b + c)(H, grad_log_q, F), axis=0) / len(y)
        return elbo, grad
    # (value, grad), autodiff_grad = jax.value_and_grad(f, has_aux=True)(phi)
    return f(phi)

@jax.jit
def online_smoothing_and_autodiff(key):
    def f(phi):
        tau = OnlineSmootherAutodiff.batch_compute(key, 
                                                    y, 
                                                    p.format_params(theta), 
                                                    phi)[0]['stats']['tau']
        return jnp.mean(tau, axis=0) / len(y)
    value, grad = jax.value_and_grad(f)(phi)
    return value, ravel_pytree(grad)[0]

@jax.jit
def online_smooth_and_autodiff_from_recursions(key):
    def f(phi):
        results = OnlineSmootherAutodiffFromRecursions.batch_compute(key, 
                                                                     y,
                                                                     p.format_params(theta), 
                                                                     phi)[0]
        return results['tau'], ravel_pytree(results['grad_tau'])[0]
    return f(phi)


@partial(jax.jit, static_argnums=1)
def oracle_smoothing_of_gradients(key, num_samples):

    formatted_phi = q.format_params(phi)
    state_seq = q.compute_state_seq(y, T-1, formatted_phi)
    

    def sample_joint(key):
        key, subkey = jax.random.split(key, 2)
        filt_params_T = q.filt_params_from_state(tree_get_idx(-1, state_seq), formatted_phi)
        sample_T = q.filt_dist.sample(subkey, filt_params_T)
        def sample_backwd(sample_tp1, key_and_states):
            key, state_t, state_tp1 = key_and_states
            backwd_params = q.backwd_params_from_states((state_t, state_tp1), formatted_phi)
            sample_t = q.backwd_kernel.sample(key, sample_tp1, backwd_params)
            return sample_t, sample_t
        
        samples_0_Tm1 = jax.lax.scan(
                                sample_backwd, 
                                init=sample_T, 
                                xs=(jax.random.split(key, T-1), 
                                    tree_droplast(state_seq), 
                                    tree_dropfirst(state_seq)),
                                reverse=True)[1]
        
        return tree_append(samples_0_Tm1, sample_T)
    
    joint_trajectories = jax.vmap(sample_joint)(random.split(key, num_samples))
    
    

    def log_joint(trajectory, phi):
        formatted_phi = q.format_params(phi)
        state_seq = q.compute_state_seq(y, T-1, formatted_phi)
        filt_params_T = q.filt_params_from_state(tree_get_idx(-1, state_seq), formatted_phi)
        log_q_T = q.filt_dist.logpdf(trajectory[-1], filt_params_T)

        def log_backwd_t_tp1(x_t, x_tp1, state_t, state_tp1):
            backwd_params = q.backwd_params_from_states((state_t, state_tp1), formatted_phi)
            return q.backwd_kernel.logpdf(x_t, x_tp1, backwd_params)

        return log_q_T + jnp.sum(jax.vmap(log_backwd_t_tp1)(trajectory[:-1], 
                                                            trajectory[1:], 
                                                            tree_droplast(state_seq),
                                                            tree_dropfirst(state_seq)))
    
    def log_last(trajectory, phi):
        formatted_phi = q.format_params(phi)
        state_seq = q.compute_state_seq(y, T-1, formatted_phi)
        filt_params_T = q.filt_params_from_state(tree_get_idx(-1, state_seq), formatted_phi)
        return q.filt_dist.logpdf(trajectory[-1], filt_params_T)
    
    gradients = jax.vmap(jax.grad(log_joint, argnums=1), in_axes=(0,None))(joint_trajectories, phi)
    gradients_last = jax.vmap(jax.grad(log_last, argnums=1), in_axes=(0,None))(joint_trajectories, phi)

    gradients = tree_map(partial(jnp.mean, axis=0), gradients)
    gradients_last = tree_map(partial(jnp.mean, axis=0), gradients_last)

    return ravel_pytree(gradients)[0], ravel_pytree(gradients_last)[0]







print('Computing oracle smoothing of gradients...')
oracle_G, oracle_G_last = oracle_smoothing_of_gradients(key, 100000)

print(oracle_G.mean())

print(oracle_G_last.mean())
#%%


# # 
# print((oracle_G - oracle_G_last).sum())

# #%%
# # offline_G = jax.vmap(oracle_smoothing_of_gradients, 
# #                      in_axes=(0,None))(jax.random.split(key, num_runs), 
# #                                                                       num_samples)

# # print('Computing online smoothing of gradients...')
# # online_G = jax.vmap(lambda key: online_smoothing_value_and_grad(key)[-1])(jax.random.split(key, num_runs))

# # print('Computing errors...')

# # similarities_online = jax.vmap(utils.cosine_similarity, 
# #                                in_axes=(None,0))(oracle_value, online_G)

# # similarities_offline = jax.vmap(utils.cosine_similarity, 
# #                                 in_axes=(None,0))(oracle_value, offline_G)

# # df = pd.DataFrame(jnp.array([
# #                         similarities_online,
# #                         similarities_offline]).T,
# #                         columns=['Offline similarity', 
# #                                  'Online similarity'])


# df.plot(kind='box')
#%%



print('Computing oracle...')
offline_values, offline_grads = offline_smoothing_value_and_grad(key)

print('Computing online...')
online_values, online_grads = jax.vmap(online_smoothing_value_and_grad)(jax.random.split(key, num_runs))

print('Computing online via autodiff...')
online_values_autodiff, online_grads_autodiff = jax.vmap(online_smoothing_and_autodiff)(jax.random.split(key, num_runs))

# print('Computing online via autodiff recursions...')
# online_values_autodiff_recursions, online_grads_autodiff_recursions = online_smooth_and_autodiff_from_recursions(key)




print('Offline oracle:', offline_values)
print('Online 3-PaRIS:', online_values)
print('Online autodiff:', online_values_autodiff)
# print('Online autodiff recursions:', online_values_autodiff_recursions)

offline_similarities = jax.vmap(utils.cosine_similarity, in_axes=(None,0))(offline_grads, online_grads)
online_similarities = jax.vmap(utils.cosine_similarity, in_axes=(None,0))(offline_grads, online_grads_autodiff)


pd.DataFrame(jnp.array([offline_similarities, online_similarities]).T, 
             columns=['Online', 'Online autodiff']).plot(kind='box')

plt.savefig('New online gradient method T=1')
#%%

# print('Diff elbo oracle and offline:', oracle_elbo_value - offline_elbo_value)
# print('Similarity grad oracle and offline:', utils.cosine_similarity(oracle_elbo_grad_value, offline_elbo_grad_value))

# # print('Diff elbo oracle and online:',oracle_elbo_value - online_elbo_value)

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






