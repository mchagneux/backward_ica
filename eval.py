#%%
import jax, jax.numpy as jnp
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_platform_name', 'cpu')


import pandas
import matplotlib.pyplot as plt
from src.variational import get_variational_model, NeuralBackwardSmoother
from src.stats.hmm import get_generative_model
from src.utils.misc import *
import os 
path = 'experiments/p_chaotic_rnn/2023_05_11__10_08_45'
num_smoothing_samples = 1000

key = jax.random.PRNGKey(0)
dummy_key = key

p_args = load_args('args', path)
p = get_generative_model(p_args)
theta_star = load_params('theta_star', path)


x = jnp.load(os.path.join(path, 'state_seqs.npy'))[0]
y = jnp.load(os.path.join(path, 'obs_seqs.npy'))[0]


models = ['linear__offline_autodiff_on_backward', 
          'johnson_backward__offline_autodiff_on_backward',
          'neural_backward__offline_autodiff_on_backward']

def eval_model(model):
    model_path = os.path.join(path, model)
    q_args = load_args('args', model_path)
    q_args.state_dim, q_args.obs_dim = p_args.state_dim, p_args.obs_dim
    q = get_variational_model(q_args, p)
    phi = load_params('phi', model_path)
    
    if isinstance(q, NeuralBackwardSmoother):
        return q.smooth_seq(key, y, phi, num_smoothing_samples)

    return q.smooth_seq(y, phi)




timesteps = jnp.arange(len(x))
for model in models: 
    fig, axes = plt.subplots(p_args.state_dim, 1, figsize=(15,10))
    plt.autoscale(True)
    plt.tight_layout()
    plt.suptitle(model)
    smoothed_means, smoothed_vars = eval_model(model)

    for dim in range(p_args.state_dim):
        ax = axes[dim]
        ax.scatter(timesteps, smoothed_means[:,dim], label='Smoothed')
        ax.scatter(timesteps, x[:,dim], label='True')
        ax.legend()
#%%


