#%%

import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_platform_name', 'cpu')
from jax.flatten_util import ravel_pytree
import haiku as hk
from src.online_smoothing import OnlineELBOScoreTruncatedGradients
from src.stats.hmm import LinearGaussianHMM, NonLinearHMM
from src.variational.sequential_models import JohnsonBackward
from src.utils.misc import exp_and_normalize, get_defaults, tree_get_strides
import matplotlib.pyplot as plt 
import blackjax
from argparse import Namespace



key = jax.random.PRNGKey(0)
args = Namespace()

args.model = 'chaotic_rnn'
args.load_from = ''
args.loaded_seq = False 
args.state_dim, args.obs_dim = 5,5
args.seq_length = 500
num_particles = 10
h = 1e-4 
num_steps = 100_000

args = get_defaults(args)

p = NonLinearHMM.chaotic_rnn(args)


key, key_theta, key_sampling = jax.random.split(key, 3)
unformatted_theta = p.get_random_params(key_theta)
theta = p.format_params(unformatted_theta)
x_true, y = p.sample_seq(key_sampling, 
                         unformatted_theta, 
                         args.seq_length)


def l_theta(x):
  init_term = p.prior_dist.logpdf(x[0], theta.prior)
  emission_terms = jax.vmap(p.emission_kernel.logpdf, in_axes=(0,0,None))(y, 
                                                                          x,
                                                                          theta.emission)
  
  transition_terms = jax.vmap(p.transition_kernel.logpdf, in_axes=(0,0,None))(x[1:], 
                                                                              x[:-1], 
                                                                              theta.transition)
  return init_term + jnp.sum(emission_terms) + jnp.sum(transition_terms)


noise = lambda key:jax.random.normal(key, shape=(num_particles, 
                                                 args.seq_length, 
                                                 args.state_dim))


key, key_init = jax.random.split(key, 2)


x_init = jax.vmap(p.sample_seq, in_axes=(0,None,None))(jax.random.split(key_init, 
                                                                        num_particles), 
                                                      unformatted_theta, 
                                                      args.seq_length)[0]

def _step(x, key):
  grad_log_l = jax.vmap(jax.grad(l_theta))(x)
  x += h*grad_log_l + jnp.sqrt(2*h)*noise(key)
  return x, None

x_end = jax.lax.scan(
                    _step, 
                    init=x_init, 
                    xs=jax.random.split(key, num_steps))[0]


x_pred = jnp.mean(x_end, 
                  axis=0)

# mala = blackjax.mala(l_theta, h)

# def _step_mala(prev_state, key):
#   new_state, _ = mala.step(key, prev_state)
#   return new_state, None

# x_end = jax.vmap(lambda key, x: jax.lax.scan(_step_mala, 
#                                           init=mala.init(x),
#                                           xs=jax.random.split(key, num_steps))[0],)(jax.random.split(key, num_particles), 
#                                                                                    x_end_ula).position

# #%%
# x_pred = jnp.mean(x_end, axis=0)

#%%
dims = args.state_dim
fig, axes = plt.subplots(dims, 1, figsize=(10, 20))
for dim in range(dims):
  axes[dim].plot(x_true[:,dim], c='red', label='True')
  axes[dim].plot(x_pred[:,dim], c='green', label='Pred')
plt.legend()
print('RMSE:',jnp.mean(jnp.sqrt(jnp.mean((x_pred-x_true)**2, axis=-1)), axis=0))
#%%
# fig, axes = plt.subplots(dims, 1)
# for dim in range(dims):
#%%



# print(F(x,z) - G(x,z))
# def G(x, z):
#     u = jax.vmap(jax.grad(g), in_axes = (None,0))(x,z)
#     v = jax.grad(h)(x)
#     return jnp.mean(jax.vmap(lambda x,y:x*y, in_axes=(0,None))(u,v))





# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):

# print(F(x,z))
# print(G(x,z))

# x, z = 2., 89.
# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
#     F(x,z).block_until_ready()

# xs = jnp.linspace(0,1,1000)
# zs = jnp.linspace(3,6,1000)

# from time import time 
# timings_F = []
# timings_G = []
# for x,z in zip(xs, zs):
#     time0 = time()
#     F(x,z).block_until_ready()
#     timings_F.append(time() - time0)
#     time0 = time()
#     G(x,z).block_until_ready()
#     timings_G.append(time() - time0)

# print(jnp.mean(jnp.array(timings_F)))
# print(jnp.mean(jnp.array(timings_G)))

path = 'experiments/p_chaotic_rnn/2023_06_29__17_53_30'
from src.stats.hmm import get_generative_model

#%%
# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):

# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
#     F(5.,jax.random.uniform(jax.random.PRNGKey(9), shape=(1000000,))).block_until_ready()
#%%
# x, y = p.sample_seq(key_sample, theta, seq_length)

# init_state = q.init_state(y[0], 
#                           q.format_params(unformatted_phi))



# formatted_phi, format_vjp = jax.vjp(q.format_params, unformatted_phi)

# def x_phi(formatted_phi):
#     state = q.new_state(y[1], init_state, formatted_phi)
#     filt_params = q.filt_params_from_state(state, formatted_phi)
#     x = q.filt_dist.sample(key, filt_params)
#     return x

# x, x_vjp = jax.vjp(x_phi, formatted_phi)


# def g_x(x):
#     return p.emission_kernel.logpdf(y[1], x, p.format_params(theta).emission)

# def g_phi(unformatted_phi):
#     x = x_phi(q.format_params(unformatted_phi))
#     return g_x(x)

# grad_g_wrt_unformatted_phi = jax.grad(g_phi)(unformatted_phi)

# grad_g_wrt_x = jax.grad(g_x)(x)

# grad_g_wrt_formatted_phi = x_vjp(grad_g_wrt_x)[0]
# grad_g_wrt_unformatted_phi_2 = format_vjp(grad_g_wrt_formatted_phi)[0]
#%%

# x = jnp.arange(0,10).astype(jnp.float64)



# squared_x, square_vjp = jax.vjp(lambda x: x**2, x)



# grad_sum_of_squared_combined = jax.grad(lambda x: jnp.sum(x**2))(x)

# grad_wrt_x = square_vjp(jax.grad(jnp.sum)(squared_x))
#%%





# def f(unformatted_phi):
#     return q.format_params(unformatted_phi)


# def g(formatted_phi):
#     return jnp.sum(formatted_phi.prior.mean)

# formatted_phi, f_vjp = jax.vjp(f, phi)
# grad_g_wrt_formatted_phi = jax.grad(g)(q.format_params(phi))


# grad_g_wrt_unformatted_phi = f_vjp(grad_g_wrt_formatted_phi)
#%%
# def g(unformatted_phi):
#     return jnp.sum(unformatted_phi.prior.mean) 

# grad_f = jax.grad(g)(phi)

# grad_f_times_grad_format = jax.tree_map(lambda x,y: 
#                                         x*y, 
#                                         f_vjp, 
#                                         grad_f)

#%%

# num_samples = 100
# elbo_options = {'variance_reduction':True, 
#                 'bptt_depth':1, 
#                 'paris':True,
#                 'normalizer':exp_and_normalize,
#                 'true_online':True,
#                 'mcmc':False}
# elbo = OnlineELBOScoreTruncatedGradients(p, q, num_samples, **elbo_options)
# # y = elbo.preprocess(y)
# carry = elbo.init_carry(phi)
# carry['theta'] = p.format_params(theta)
# input_0 = {'t':0, 
#            'T':0, 
#            'key':jax.random.PRNGKey(0), 
#            'phi':phi, 
#            'ys_bptt':y[0]}

# input_1 = {'t':1, 
#            'T':1, 
#            'key':jax.random.PRNGKey(0), 
#            'phi':phi, 
#            'ys_bptt':y[1]}

# carry, _ = jax.jit(elbo._init)(carry, input_0)
# carry['theta'] = p.format_params(theta)

# def f(carry, input):
#     new_carry, aux = elbo._update(carry, input)
#     elbo_value, neg_grad = elbo.postprocess(new_carry)
#     return neg_grad




# def f(unformatted_phi, y, state_t, key_t):
#     def _log_q_t_bar(unformatted_phi, key_t):
#         phi = q.format_params(unformatted_phi)
#         new_state = q.new_state(y, state_t, phi)
#         params_q_t = q.filt_params_from_state(new_state, phi)
#         x_t = q.filt_dist.sample(key_t, params_q_t)
#         return q.filt_dist.logpdf(x_t, params_q_t)

#     def _p_theta_emission_bar(unformatted_phi, key_t):
#         phi = q.format_params(unformatted_phi)
#         new_state = q.new_state(y, state_t, phi)
#         params_q_t = q.filt_params_from_state(new_state, phi)
#         x_t = q.filt_dist.sample(key_t, params_q_t)
#         return p.emission_kernel.logpdf(y, x_t, p.format_params(theta).emission)
    
#     log_q_t_bar, grad_log_q_t_bar = jax.value_and_grad(_log_q_t_bar)(unformatted_phi, key_t)
#     log_g_theta_bar, grad_log_g_theta_bar = jax.value_and_grad(_p_theta_emission_bar)(unformatted_phi, key_t)

#     return (log_q_t_bar, log_g_theta_bar), jax.tree_map(lambda x,y: x+y, grad_log_q_t_bar, grad_log_g_theta_bar)


# state_t = q.init_state(y[0], q.format_params(phi))

# with open('lowered_jit.txt', 'w') as file: 
#     file.write(jax.jit(f).lower(phi, y[1], state_t, key).compile().as_text())

# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
#     neg_grad = jax.jit(f)(carry, input_1)
#     print(neg_grad)
#%%