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

experiment_path = 'experiments/p_chaotic_rnn/2023_04_03__15_03_01/johnson_backward' 
p_and_data_path = os.path.split(experiment_path)[0]
num_samples = 1000
num_samples_oracle = 10000
T = 2
utils.enable_x64(True)
jax.config.update('jax_disable_jit', False)

base_key = jax.random.PRNGKey(0)
args_p = utils.load_args('args',p_and_data_path)
args_q = utils.load_args('args', experiment_path)
args_q.state_dim, args_q.obs_dim = args_p.state_dim, args_p.obs_dim

p = get_generative_model(args_p)
q = get_variational_model(args_q)


theta = utils.load_params('theta_star', p_and_data_path)
with open(os.path.join(experiment_path, 'data'), 'rb') as f: 
    phi = dill.load(f)[-1]

y = jnp.load(os.path.join(p_and_data_path, 'obs_seqs.npy')).squeeze()[:T]

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
        return results['H'], ravel_pytree(results['F'])[0], ravel_pytree(results['G'])[0]
    # (value, grad), autodiff_grad = jax.value_and_grad(f, has_aux=True)(phi)
    return f(phi)

@jax.jit
def online_smoothing_and_autodiff(key):
    def f(phi):
        return OnlineSmootherAutodiff.batch_compute(key, y, p.format_params(theta), phi)[0]['tau']
    value, grad = jax.value_and_grad(f)(phi)
    return value, ravel_pytree(grad)[0]

@jax.jit
def online_smooth_and_autodiff_from_recursions(key):
    def f(phi):
        results = OnlineSmootherAutodiffFromRecursions.batch_compute(key, y, p.format_params(theta), phi)[0]
        return results['tau'], ravel_pytree(results['grad_tau'])[0]
    return f(phi)


@jax.jit
def oracle_smoothing_of_gradients(key):

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
    
    joint_trajectories = jax.vmap(sample_joint)(random.split(key, num_samples_oracle))
    

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
    
    gradients = jax.vmap(jax.grad(log_joint, argnums=1), in_axes=(0,None))(joint_trajectories, phi)
    
    gradients = tree_map(partial(jnp.mean, axis=0), gradients)

    return ravel_pytree(gradients)[0]





print('Computing oracle smoothing of gradients...')
oracle_G = oracle_smoothing_of_gradients(key)

print('Computing online smoothing of gradients...')
online_G = online_smoothing_value_and_grad(key)[-1]

print(utils.cosine_similarity(oracle_G, online_G))

# print('Computing oracle...')
# offline_values, offline_grads = offline_smoothing_value_and_grad(key)

# print('Computing online...')
# online_values, online_grads = online_smoothing_value_and_grad(key)

# print('Computing online via autodiff...')
# online_values_autodiff, online_grads_autodiff = online_smoothing_and_autodiff(key)

# print('Computing online via autodiff recursions...')
# online_values_autodiff_recursions, online_grads_autodiff_recursions = online_smooth_and_autodiff_from_recursions(key)

# print('Offline oracle:', offline_values)
# print('Online 3-PaRIS:', online_values)
# print('Online autodiff:', online_values_autodiff)
# print('Online autodiff recursions:', online_values_autodiff_recursions)

# print('Similarity offline and online 3-PaRIS:', utils.cosine_similarity(
#                                                         offline_grads, 
#                                                         online_grads))

# print('Similarity offline and online autodiff:',utils.cosine_similarity(
#                                                         offline_grads, 
#                                                         online_grads_autodiff))

# print('Similarity offline and online autodiff recursions:',utils.cosine_similarity(
#                                                         offline_grads, 
#                                                         online_grads_autodiff_recursions))

# print('Similarity online autodiff and online autodiff recursions:',utils.cosine_similarity(
#                                                         online_grads_autodiff, 
#                                                         online_grads_autodiff_recursions))


# print(jnp.sum(jnp.linalg.norm(pykalman_smooth_seq(y, q.format_params(phi))[0], axis=-1)))

# from pykalman.standard import KalmanFilter
# def pykalman_smooth_seq(obs_seq, hmm_params):

#     engine = KalmanFilter(transition_matrices=hmm_params.transition.map.w, 
#                         observation_matrices=hmm_params.emission.map.w,
#                         transition_covariance=hmm_params.transition.noise.scale.cov,
#                         observation_covariance=hmm_params.emission.noise.scale.cov,
#                         transition_offsets=hmm_params.transition.map.b,
#                         observation_offsets=hmm_params.emission.map.b,
#                         initial_state_mean=hmm_params.prior.mean,
#                         initial_state_covariance=hmm_params.prior.scale.cov)
                        
#     return engine.smooth(obs_seq)

        
# oracle_elbo_value, oracle_elbo_grad_value = oracle_elbo_and_grad()
# offline_elbo_value, offline_elbo_grad_value = offline_elbo_and_grad(jax.random.PRNGKey(1))
# # online_elbo_value, online_elbo_grad_value = online_elbo_and_grad(key)



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






