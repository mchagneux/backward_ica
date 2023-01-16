from backward_ica.stats.hmm import LinearGaussianHMM
from backward_ica.online_smoothing import OnlineNormalizedISELBO, ELBOGradsReparam
from backward_ica.offline_smoothing import LinearGaussianELBO, GeneralBackwardELBO
import jax, jax.numpy as jnp
from functools import partial
from backward_ica.utils import enable_x64
import os 
from jax.tree_util import tree_map
d_x = 1
d_y = 1
import pandas as pd 
import matplotlib.pyplot as plt 

enable_x64(True)




p = LinearGaussianHMM(
                    state_dim=d_x,
                    obs_dim=d_y, 
                    transition_matrix_conditionning='diagonal',
                    range_transition_map_params=(0.1,1),
                    transition_bias=False,
                    emission_bias=False)

q = LinearGaussianHMM(state_dim=d_x,
                    obs_dim=d_y, 
                    transition_matrix_conditionning='diagonal',
                    range_transition_map_params=(0.1,1),
                    transition_bias=False,
                    emission_bias=False)




num_samples = 100
num_replicas = 20
seq_length = 50



key = jax.random.PRNGKey(5)


path = 'experiments/online/grad_computation_eval'

def experiment(key, exp_id):

    oracle_elbo = LinearGaussianELBO(p, q)
    online_elbo = OnlineNormalizedISELBO(p, q, num_samples)
    online_elbo_2 = ELBOGradsReparam(p, q, num_samples)
    offline_elbo = GeneralBackwardELBO(p, q, num_samples)

    key, key_theta, key_phi = jax.random.split(key, 3)

    theta = p.get_random_params(key_theta)
    phi = q.get_random_params(key_phi)


    def oracle_elbo_and_grad_func(obs_seq):

        elbo, grad_elbo = jax.value_and_grad(lambda obs_seq, phi: oracle_elbo(obs_seq, 
                                                    len(obs_seq)-1, 
                                                    p.format_params(theta), 
                                                    q.format_params(phi)), 
                                                    argnums=1)(obs_seq, phi)
        return elbo, jax.flatten_util.ravel_pytree(grad_elbo)[0]



    def offline_elbo_and_grad_func(key, obs_seq):
        elbo, grad_elbo = jax.value_and_grad(lambda obs_seq, phi: offline_elbo(key, obs_seq, 
                                                    len(obs_seq)-1, 
                                                    p.format_params(theta), 
                                                    q.format_params(phi)), 
                                                    argnums=1)(obs_seq, phi)

        return elbo, jax.flatten_util.ravel_pytree(grad_elbo)[0]




    def online_elbo_and_grad_func(key, obs_seq):
        elbo, grad_elbo = jax.value_and_grad(lambda obs_seq, phi: online_elbo.batch_compute(key, obs_seq, 
                                                    p.format_params(theta), 
                                                    phi)['tau'], 
                                                    argnums=1)(obs_seq, phi)

        return elbo, jax.flatten_util.ravel_pytree(grad_elbo)[0]


    def online_elbo_and_grad_func_2(key, obs_seq):
        stats = online_elbo_2.batch_compute(key, obs_seq, 
                                                    p.format_params(theta), 
                                                    phi)
        elbo, grad_elbo = stats['Omega'], stats['jac_Omega']

        return elbo, jax.flatten_util.ravel_pytree(grad_elbo)[0]





    key, key_seq = jax.random.split(key, 2)

    state_seq, obs_seq = p.sample_seq(key_seq, theta, seq_length)

    oracle_elbo, oracle_grad_elbo = jax.jit(oracle_elbo_and_grad_func)(obs_seq)
    norm_oracle_grad = jnp.linalg.norm(oracle_grad_elbo, ord=1)
    # print('Oracle ELBO', oracle_elbo)
    # print('Oracle grad norm ELBO', oracle_grad_elbo)


    # print('-----')
    offline_elbos, offline_grads_elbo = jax.jit(jax.vmap(offline_elbo_and_grad_func, in_axes=(0,None)))(jax.random.split(key, num_replicas), obs_seq)
    offline_elbo_errors = (offline_elbos - oracle_elbo) / jnp.abs(oracle_elbo)
    offline_grad_elbo_errors = (jnp.linalg.norm(offline_grads_elbo, axis=1, ord=1) - norm_oracle_grad) / norm_oracle_grad
    # print('Offline ELBOs',offline_elbos)
    # print('Offline grad norm ELBO', offline_grads_elbo)

    # print('-----')

    online_elbos, online_grads_elbo = jax.jit(jax.vmap(online_elbo_and_grad_func, in_axes=(0,None)))(jax.random.split(key, num_replicas), obs_seq)
    online_elbo_errors = (online_elbos - oracle_elbo) / jnp.abs(oracle_elbo)
    online_grad_elbo_errors = (jnp.linalg.norm(online_grads_elbo, axis=1, ord=1) - norm_oracle_grad) / norm_oracle_grad


    online_elbos_2, online_grads_elbo_2 = jax.jit(jax.vmap(online_elbo_and_grad_func_2, in_axes=(0,None)))(jax.random.split(key, num_replicas), obs_seq)
    online_elbo_errors_2 = (online_elbos_2 - oracle_elbo) / jnp.abs(oracle_elbo)
    online_grad_elbo_errors_2 = (jnp.linalg.norm(online_grads_elbo_2, axis=1, ord=1) - norm_oracle_grad) / norm_oracle_grad

    elbo_errors = pd.DataFrame(jnp.array([offline_elbo_errors, online_elbo_errors, online_elbo_errors_2]).T, columns=['Offline', 'Online', 'Online 2'])
    grad_elbo_errors = pd.DataFrame(jnp.array([offline_grad_elbo_errors, online_grad_elbo_errors, online_grad_elbo_errors_2]).T, columns=['Offline', 'Online', 'Online 2'])

    elbo_errors.plot(kind='box')
    plt.savefig(os.path.join(path, f'elbo_errors_{exp_id}'))

    grad_elbo_errors.plot(kind='box')
    plt.savefig(os.path.join(path, f'grad_elbo_errors_{exp_id}'))

for exp_id in range(5):
    key, subkey = jax.random.split(key, 2)
    experiment(subkey, exp_id)

# elbo_errors.plot(kind='box')
# elbo, grad_elbo = oracle_elbo_and_grad_func(obs_seq)

# grad, _ = jax.flatten_util.ravel_pytree(grad_elbo)

# print(grad)


# theta = {0: jnp.array([0.2, 0.4]), 1:  jnp.array([0.6, 0.9])}

# a = jnp.array([1,2])
# b = jnp.array([4,5])

# def f(theta, x):
#     return (theta[0] + theta[1]) * x, theta[0]*theta[1]

# def g(theta, x):
#     return theta[0]*(x**2) + theta[1] 

# def h(theta,x):
#     return x + (theta[0] * theta[1]) ** 2

# def F(theta, x):
#     return f(theta, x) @ (g(theta, x) + h(theta, x))





# grad_F_jax = jax.vmap(jax.grad(F, argnums=0), in_axes=(None,0))

# jac_h = jax.jacrev(h, argnums=0)


# def grad_F_custom(theta, x):

#     f_x, vjp_f_x, s_t = jax.vjp(partial(f, x=x), theta, has_aux=True)
#     g_x, vjp_g_x = jax.vjp(partial(g, x=x), theta)
#     jac_h_x = jac_h(theta, x)
#     h_x = h(theta, x)

#     temp_term = tree_map(lambda x: x.T @ f_x, jac_h_x)

#     return tree_map(lambda x,y,z,t: x+y+z+t, temp_term, vjp_f_x(h_x)[0], vjp_f_x(g_x)[0], vjp_g_x(f_x)[0]), s_t

# x = jnp.array([[1., 1.2], 
#             [3.4, 5]])


# grad_F, s_t = jax.vmap(grad_F_custom, in_axes=(None,0))(theta, x)

# print('------------')
# print(jax.jit(jax.vmap(grad_F_custom, in_axes=(None, 0))).lower(theta, x).compile().as_text())

# print(jax.vmap(my_grad, in_axes=(None, 0))(theta, x))

# x = jnp.array([2., 2.])
# y = jnp.array([3., 3.])

# grad_autodiff = jax.grad(h)

# def my_grad(x):
#     f_x, vjp_f_x = jax.vjp(f,x)
#     g_x, vjp_g_x = jax.vjp(g,x)
#     return vjp_f_x(g_x)[0] + vjp_g_x(f_x)[0]

# def my_grad_2(x):
#     return jax.jacrev(f)(x).T @ g(x) + jax.jacrev(g) @ f(x)


# def F(x):
#     return jnp.array([y * x, x**2])


# # print(F(x))
# print(jax.jacrev(F)(x)[0])


# print(jax.grad(g)(x) @ h(x) + g(x) @ jax.grad(h)(x))

# def grads_twice(key, x0, obs, init_state, theta):

#     def filt_params(theta):
#         theta = p.format_params(theta)
#         return p.filt_params_from_state(p.new_state(obs, init_state, theta), theta)

#     def backwd_params(theta):
#         theta = p.format_params(theta)
#         return p.backwd_params_from_state(p.new_state(obs, init_state, theta), theta)


#     def x1(key, theta):
#         return p.filt_dist.sample(key, filt_params(theta))
    
#     r1 = jax.grad(lambda theta, key, x: p.backwd_kernel.logpdf(
#                                                     x=x, 
#                                                     state=x1(key, theta),
#                                                     params=backwd_params(theta)))(theta, key, x0)

#     r2 = jax.grad(lambda theta, key: p.filt_dist.logpdf(
# 









# print(grads_twice(key, x0, obs, init_state, theta))

# print(jax.jit(grads_twice).lower(key, x0, obs, init_state, theta).compile().as_text())




              





    









