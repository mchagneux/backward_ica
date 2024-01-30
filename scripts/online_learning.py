#%%
import argparse 
import jax, jax.numpy as jnp
import dill 
import matplotlib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
jax.config.update('jax_platform_name', 'cpu')
# jax.config.update('jax_disable_jit', True)

from src.utils.misc import get_defaults, save_args, load_args, tree_get_slice
from src.stats.hmm import get_generative_model, HMM
from src.variational import get_variational_model
from src.training import SVITrainer
color = sns.color_palette()[1]
alpha = 0.2
matplotlib.rc('font',size=15)

#%%

def set_p_args(load, model, seq_length, n_bootstrap, n_ffbsi, exp_path, d_x, d_y):
  if load:
    p_args = load_args('p_args', exp_path)
  else: 
    os.makedirs(exp_path)
    p_args = argparse.Namespace()
    p_args.state_dim, p_args.obs_dim = d_x, d_y
    p_args.model = model
    # p_args.load_from = 'data/crnn/2023-06-09_15-46-10_Train_run'
    p_args.load_from = ''
    p_args.loaded_seq = False
    p_args.seq_length = seq_length
    p_args = get_defaults(p_args)
    save_args(p_args,'p_args',exp_path)
  p_args.num_particles = n_bootstrap
  p_args.num_smooth_particles = n_ffbsi
  return p_args

def set_q_args(load, exp_path, p_args, model):
  model_path = os.path.join(exp_path, model)
  if load: 
    q_args = load_args('q_args', model_path)
  else:
    os.makedirs(model_path, exist_ok=True)
    q_args = argparse.Namespace()
    q_args.state_dim, q_args.obs_dim = p_args.state_dim, p_args.obs_dim
    q_args.model = model
    q_args = get_defaults(q_args)
    q_args.optimizer = 'adam'
    q_args.learning_rate = 1e-3
    q_args.optim_options = 'cst'
    q_args.num_epochs = 1
    q_args.num_samples = 50
    q_args.training_mode = 'true_online,1,difference'
    q_args.elbo_mode = 'score,resampling,bptt_depth_2'

    # q_args.training_mode = f'reset,{p_args.seq_length},1'
    # q_args.elbo_mode = 'autodiff_on_batch'

    q_args.logging_type = 'basic_logging'
    save_args(q_args, 'q_args', model_path)
  return q_args

def get_sequence(key, p:HMM, theta, p_args, load, exp_path, name=''):
  
  if load: 
    xs = jnp.load(os.path.join(exp_path,f'xs_{name}.npy'))
    ys = jnp.load(os.path.join(exp_path,f'ys_{name}.npy'))
  else: 
    xs, ys = p.sample_multiple_sequences(
                                    key, 
                                    theta, 
                                    1, 
                                    p_args.seq_length, 
                                    single_split_seq=False,
                                    load_from=p_args.load_from,
                                    loaded_seq=p_args.loaded_seq)
    
    jnp.save(os.path.join(exp_path,f'xs_{name}.npy'), xs)
    jnp.save(os.path.join(exp_path,f'ys_{name}.npy'), ys)
  return xs, ys

def get_params_q(key, p, theta, p_args, q_args, data, load, exp_path):
  q = get_variational_model(q_args)

  if load:
    with open(os.path.join(exp_path, q_args.model, 'params'), 'rb') as f: 
      fitted_params = dill.load(f)

  trainer = SVITrainer(p=p,
                       theta_star=theta,
                       q=q, 
                       optimizer=q_args.optimizer,
                       learning_rate=q_args.learning_rate,
                       optim_options=q_args.optim_options,
                       num_epochs=q_args.num_epochs, 
                       seq_length=p_args.seq_length,
                       num_samples=q_args.num_samples,
                       force_full_mc=False,
                       frozen_params='',
                       num_seqs=1,
                       training_mode=q_args.training_mode,
                       elbo_mode=q_args.elbo_mode,
                       logging_type=q_args.logging_type)

  key, key_params, key_mc = jax.random.split(key, 3)
  fitted_params, elbos = trainer.fit(key_params, 
                                  key_mc, 
                                  data, 
                                  None,
                                  q_args,
                                  None)

  plt.plot(elbos.squeeze())

  with open(os.path.join(exp_path, q_args.model, 'params'), 'wb') as f: 
      dill.dump(fitted_params, f)

  return q, fitted_params

#%%

load = ''
date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
if load == '': 
  exp_path = f'experiments/online_experiments/{date}'
  load = False
else:
  exp_path = load
  load = True

p_model = 'chaotic_rnn'
seq_length = 100_000
n_bootstrap = 10
n_ffbsi = 50
d_x, d_y = 10,10
num_epochs_vi_learning = 1

key = jax.random.PRNGKey(0)
q_model = 'johnson_backward,100'

key, key_theta, key_train_seqs, key_smc, key_vi = jax.random.split(key, 5)
p_args = set_p_args(load, 
                    p_model, 
                    seq_length, 
                    n_bootstrap, 
                    n_ffbsi, 
                    exp_path, 
                    d_x, 
                    d_y)


p, theta = get_generative_model(p_args, 
                                key_theta)

xs, ys = get_sequence(key_train_seqs, p, theta, p_args, load, exp_path, '')

#%%
# smoothed_means_ula = p.ula_smoothing(key, ys[0], theta, 0.001, 5_000)
#%%
q_args = set_q_args(load, exp_path, p_args, q_model)
q, params_q = get_params_q(key_vi, p, theta, p_args, q_args, (xs, ys), load, exp_path)  
#%%
burnin = 10_000
smoothed_means = q.smooth_seq(ys[0], params_q)[0]
#%%
fig, axes = plt.subplots(d_x, 1, figsize=(20,15))
plt.autoscale(True)
for d in range(d_x):
  axes[d].plot(xs[0][burnin:,d], label='True states')
  axes[d].plot(smoothed_means[burnin:,d], label='Smoothed states')
  axes[d].legend()
#%%