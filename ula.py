#%%

import jax, jax.numpy as jnp
import numpy as np 
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_platform_name', 'gpu')

# jax.config.update('jax_platform_name', 'gpu')
from jax.flatten_util import ravel_pytree
import haiku as hk
from src.online_smoothing import OnlineELBOScoreTruncatedGradients
from src.stats.hmm import LinearGaussianHMM, NonLinearHMM, get_generative_model 
from src.variational.sequential_models import JohnsonBackward
from src.utils.misc import exp_and_normalize, get_defaults, tree_get_strides
from src.training import SVITrainer
import matplotlib.pyplot as plt 
import blackjax
from argparse import Namespace
from src.utils.misc import load_params
from tqdm import tqdm
import os
from time import time 
from functools import partial

key = jax.random.PRNGKey(0)
args = Namespace()

args.model = 'chaotic_rnn'
args.default_emission_matrix = None
args.load_from = '' #data/crnn/2022-10-18_15-28-00_Train_run'
args.loaded_seq = False
args.state_dim, args.obs_dim = 5,5
args.seq_length = 500
args.num_particles = 10_000
args.num_smooth_particles = 50
args.num_seqs = 1
svgd_kernel_base = lambda x,y: jnp.exp(-jnp.sum((x-y)**2) / 0.01)

num_ula_particles = args.num_smooth_particles
h = 1e-3
num_parametric_vi_samples = args.num_smooth_particles
num_epochs_parametric_vi = 1_000
num_ula_steps = 10_000


args = get_defaults(args)
# args.default_emission_base_scale = 0.0001
del args.default_emission_matrix
#%% 
key, key_theta, key_sampling = jax.random.split(key, 3)

# p = NonLinearHMM.chaotic_rnn(args)

p, unformatted_theta = get_generative_model(args, key_theta)
#%%
# unformatted_theta = p.get_random_params(key_theta)
theta = p.format_params(unformatted_theta)

x_true, y = p.sample_multiple_sequences(key_sampling, 
                                        unformatted_theta, 
                                        args.num_seqs, 
                                        args.seq_length,
                                        False,
                                        args.load_from,
                                        args.loaded_seq)


def plot_x_true_against_x_pred(x_pred):
  dims = args.state_dim
  fig, axes = plt.subplots(dims, 1, figsize=(15,1.5*args.state_dim))
  for dim in range(dims):
    axes[dim].plot(x_true[0][:,dim], c='red', label='True')
    axes[dim].plot(x_pred[:,dim], c='green', label='Pred')
    axes[dim].legend()
  # plt.legend()s
  rmse = jnp.mean(jnp.sqrt(jnp.mean((x_pred-x_true[0])**2, axis=-1)), axis=0)
  print('RMSE:',rmse)
  return rmse

# plot_x_true_against_x_pred(y[0])
#%%

def variational_fit(num_epochs, num_samples):

  training_mode = f'reset,{args.seq_length},1'
  learning_rate = 1e-2
  elbo_mode = 'autodiff_on_backward'
  optim_options = 'cst'
  batch_size = 1
  optimizer = 'adam'
  q = JohnsonBackward(args.state_dim, 
                      args.obs_dim, 
                      'diagonal', 
                      (0.9,0.99), 
                      False, 
                      (100,), 
                      False)

  q_trainer = SVITrainer(p, 
                        unformatted_theta,
                        q, 
                        optimizer,
                        learning_rate,
                        optim_options,
                        num_epochs,
                        batch_size,
                        args.seq_length,
                        num_samples,
                        False,
                        '',
                        training_mode,
                        elbo_mode)
  progress_bar = tqdm(total=num_epochs,
                      desc='ELBO')
  data = (x_true, y)
  key = jax.random.PRNGKey(0)
  key_params, key_montecarlo = jax.random.split(key, 2)
  for params, elbo, _ in q_trainer.fit(key_params, key_montecarlo, data, None, args):
    progress_bar.update(1)
    progress_bar.set_postfix({'ELBO':elbo})
  progress_bar.close()

  x_pred = q.smooth_seq(y[0], params)[0]

  return x_pred

def l_theta_vmapped(x):
  init_term = p.prior_dist.logpdf(x[0], theta.prior)
  emission_terms = jax.vmap(p.emission_kernel.logpdf, 
                            in_axes=(0,0,None))(
                                              y[0], 
                                              x,
                                              theta.emission)
  
  transition_terms = jax.vmap(p.transition_kernel.logpdf, 
                              in_axes=(0,0,None))(x[1:], 
                                                  x[:-1], 
                                                  theta.transition)
  return init_term + jnp.sum(emission_terms) + jnp.sum(transition_terms)

def l_theta_recursive(x):
  init_term = p.prior_dist.logpdf(x[0], theta.prior) \
              + p.emission_kernel.logpdf(y[0][0], x[0], theta.emission)
  
  def _step(carry, new_terms):
    x_prev, current_value = carry
    x, y = new_terms 
    new_value = current_value \
              + p.transition_kernel.logpdf(x, x_prev, theta.transition) \
              + p.emission_kernel.logpdf(y, x, theta.emission)
    return (x, new_value), None
  
  return jax.lax.scan(_step, init=(x[0], init_term), xs=(x[1:], y[0][1:]))[0][1]

l_theta = l_theta_vmapped

def ula_fit(key, 
            num_steps, 
            num_particles):

  key, key_init = jax.random.split(key, 2)
  keys = jax.random.split(key, num_steps)

  x_init = jax.jit(jax.vmap(p.sample_prior, in_axes=(0,None,None)), static_argnums=2)(
                                        jax.random.split(key_init, num_particles), 
                                        unformatted_theta, 
                                        args.seq_length)
  

  def _step(x, key):
    grad_log_l = jax.vmap(jax.grad(l_theta))(x)
    x += h*grad_log_l + jnp.sqrt(2*h)*jax.random.normal(key, shape=(num_particles, 
                                                                  args.seq_length, 
                                                                  args.state_dim))
    return x, None

  

  x_end = jax.lax.scan(
                      _step, 
                      init=x_init, 
                      xs=keys)[0]


  x_pred = jnp.mean(x_end, 
                    axis=0)

  return x_pred


def svgd_fit(num_steps, num_particles):



  x_init = jax.jit(jax.vmap(p.sample_prior, in_axes=(0,None,None)), static_argnums=2)(
                                        jax.random.split(key, num_particles), 
                                        unformatted_theta, 
                                        args.seq_length)

  def _step(x, dummy_in):

    def _particle_update(x_i):
      def _sum_term(x_i, x_j):
        return jax.grad(l_theta)(x_j)*svgd_kernel_base(x_j, x_i) + jax.grad(svgd_kernel_base)(x_j, x_i)
      return x_i + h*jnp.mean(jax.vmap(_sum_term, in_axes=(None, 0))(x_i, x))
      
    return jax.vmap(_particle_update)(x), None
  
  x_end = jax.lax.scan(
                      _step, 
                      init=x_init, 
                      xs=None,
                      length=num_steps)[0]


  x_pred = jnp.mean(x_end, 
                    axis=0)

  return x_pred

  

# variational_fit(num_ula_steps, num_ula_particles)
#%%


# # x_pred = variational_fit(num_epochs_parametric_vi, 
# #                          num_parametric_vi_samples)
# # rmse = plot_x_true_against_x_pred(x_pred)
# # # plt.savefig('parameterized_vi_example.pdf', format='pdf')
# print('Precompiling ULA loop...')
# _ = ula_fit(key, 2, num_ula_particles)
# print('Running ULA loop...')
# time0 = time()
# x_pred = ula_fit(key, 
#                  num_ula_steps, 
#                  num_ula_particles)
# x_pred.block_until_ready()
# print('ULA time:', time() - time0)
# rmse = plot_x_true_against_x_pred(x_pred)
#%%

# print('Precompiling svgd loop...')
# _ = svgd_fit(2, num_ula_particles)
# print('Running svgd loop...')
# time0 = time()
# x_pred = svgd_fit(num_ula_steps, 
#                  num_ula_particles)
# x_pred.block_until_ready()
# print('SVGD time:', time() - time0)
# rmse = plot_x_true_against_x_pred(x_pred)

probs, positions = p.filt_seq(key, y[0], unformatted_theta)
x_pred = jax.vmap(lambda w,x: jnp.sum(jax.vmap(lambda w,x:w*x)(w,x), axis=0))(probs, positions)
# x_pred = p.smooth_seq(key, y[0], unformatted_theta)[0]
rmse = plot_x_true_against_x_pred(x_pred)

#%%
#%%
# x_pred = jax.vmap(lambda w,x: jnp.sum(jax.vmap(lambda w,x:w*x)(w,x), axis=0))(probs, positions)
# rmse = plot_x_true_against_x_pred(x_pred)


#%%
#%%
# plt.autoscale(True)
# plt.tight_layout()
# plt.suptitle(f'RMSE:{rmse:.3f}')
# plt.savefig('ula_example.pdf', format='pdf')
#%%



