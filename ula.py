#%%

import jax, jax.numpy as jnp
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


key = jax.random.PRNGKey(0)
args = Namespace()

args.model = 'chaotic_rnn_with_nonlinear_emission'
args.load_from = '' #data/crnn/2022-10-18_15-28-00_Train_run'
args.loaded_seq = False
args.state_dim, args.obs_dim = 64,1000
args.seq_length = 10_000
args.num_particles = 100
args.num_smooth_particles = 2
args.num_seqs = 1

num_ula_particles = 50
h = 1e-3
num_ula_steps = 1000


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
  plt.legend()
  print('RMSE:',jnp.mean(jnp.sqrt(jnp.mean((x_pred-x_true[0])**2, axis=-1)), axis=0))

# plot_x_true_against_x_pred(y[0])
#%%

def variational_fit(num_epochs, num_samples):

  training_mode = f'reset,{args.seq_length},1'
  learning_rate = 1e-2
  elbo_mode = 'autodiff_on_backward'
  optim_options = 'cst'
  num_epochs = 2000
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

  plot_x_true_against_x_pred(x_pred)


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


def ula_fit(key, num_steps, num_particles):

  noise = lambda key:jax.random.normal(key, shape=(num_ula_particles, 
                                                  args.seq_length, 
                                                  args.state_dim))


  key, key_init = jax.random.split(key, 2)

  @jax.jit
  def _step(x, key):
    grad_log_l = jax.vmap(jax.grad(l_theta))(x)
    x += h*grad_log_l + jnp.sqrt(2*h)*noise(key)
    return x, None

  x_init = jax.vmap(p.sample_prior, in_axes=(0,None,None))(
                                        jax.random.split(key_init, num_particles), 
                                        unformatted_theta, 
                                        args.seq_length)
  
  
  keys = jax.random.split(key, num_steps)

  # print('Pre-compiling ULA loop...')
  # _ = _step(x_init, key)

  # print('Running ULA loop...')
  # x = x_init
  # for key in tqdm(keys):
  #   x = _step(x, key)
  # x_end = x
  # x_pred = jnp.mean(x_end, 
  #                   axis=0)

  from time import time 
  print('Pre-compiling ULA loop...')
  x_end = jax.lax.scan(
                      _step, 
                      init=x_init, 
                      xs=jax.random.split(key, 2))[0]
  print('Running ULA...')
  time0 = time()
  x_end = jax.lax.scan(
                      _step, 
                      init=x_init, 
                      xs=keys)[0]


  x_pred = jnp.mean(x_end, 
                    axis=0)
  x_pred.block_until_ready()
  print('ULA time:', time() - time0)

  plot_x_true_against_x_pred(x_pred)



# variational_fit(num_ula_steps, num_ula_particles)
#%%
ula_fit(key, num_ula_steps, num_ula_particles)
#%%



