#%%
import jax, jax.numpy as jnp
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', False)


import pandas
import matplotlib.pyplot as plt
from src.variational import get_variational_model, NeuralBackwardSmoother
from src.stats.hmm import get_generative_model, LinearGaussianHMM
from src.utils.misc import *
import os 


key = jax.random.PRNGKey(0)
num_seqs = 10
experiment_path = 'experiments/p_chaotic_rnn/2023_07_07__18_19_00'
model_path = 'johnson_backward,200.200.adam,1e-3,cst.true_online,1,difference.score,paris,bptt_depth_2.gpu.basic_logging'
full_model_path = os.path.join(experiment_path, model_path)
p_args = load_args('args', experiment_path)
seq_length = p_args.seq_length

p = get_generative_model(p_args)
theta_star = load_params('theta_star', experiment_path)
xs, ys = p.sample_multiple_sequences(key, theta_star, num_seqs, seq_length)

q_args = load_args('args', full_model_path)
q_args.state_dim, q_args.obs_dim = p_args.state_dim, p_args.obs_dim
q = get_variational_model(q_args, p)
phi = load_params('phi', full_model_path)
smoothed_xs = jax.jit(jax.vmap(lambda y: q.smooth_seq(y, phi)[0]))(ys)
filt_xs = jax.jit(jax.vmap(lambda y: q.filt_seq(y, phi)[0]))(ys)
def rmse_on_seq(x_true, x_pred):
    return jnp.mean(jnp.sqrt(jnp.mean((x_true-x_pred)**2, axis=-1)))

rmses_smooth = jax.vmap(rmse_on_seq)(xs, smoothed_xs)
rmses_filt = jax.vmap(rmse_on_seq)(xs, filt_xs)

print('Smoothing mean RMSE:', rmses_smooth.mean())
print('Smoothing std RMSE:', rmses_smooth.std())
print('---')
print('Filtering mean RMSE:', rmses_filt.mean())
print('Filtering std RMSE:', rmses_filt.std())
#%%
