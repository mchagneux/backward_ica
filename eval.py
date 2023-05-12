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
path = 'experiments/p_chaotic_rnn/2023_05_12__15_58_38'
num_smoothing_samples = 1000

key = jax.random.PRNGKey(0)
dummy_key = key

p_args = load_args('args', path)
p = get_generative_model(p_args)
theta_star = load_params('theta_star', path)


x = jnp.load(os.path.join(path, 'state_seqs.npy'))[0]
y = jnp.load(os.path.join(path, 'obs_seqs.npy'))[0]
T = len(y) - 1 

models = ['johnson_backward__offline_autodiff_on_backward',
          'johnson_backward__offline_score_variance_reduction_bptt_depth_1',
          'johnson_backward__offline_score_variance_reduction_bptt_depth_5']

def eval_model(model):
    model_path = os.path.join(path, model)
    q_args = load_args('args', model_path)
    q_args.state_dim, q_args.obs_dim = p_args.state_dim, p_args.obs_dim
    q = get_variational_model(q_args, p)
    phi = load_params('phi', model_path)
    

    def filt_seq():
        state_seq = q.compute_state_seq(y, T, q.format_params(phi))
        def get_mean_and_cov(state):
            filt_params = q.filt_params_from_state(state, phi)
            return filt_params.mean, filt_params.scale.cov

        return jax.vmap(get_mean_and_cov)(state_seq)
    
    if isinstance(q, NeuralBackwardSmoother):
        means_smooth, covs_smooth = q.smooth_seq(key, y, phi, num_smoothing_samples)
    else: 
        means_smooth, covs_smooth = q.smooth_seq(y, phi)
        covs_smooth = jax.vmap(jnp.diagonal)(covs_smooth)

    means_filt, covs_filt = filt_seq()
    covs_filt = jax.vmap(jnp.diag)(covs_filt)

    return (means_smooth, jnp.sqrt(covs_smooth)), (means_filt, jnp.sqrt(covs_filt))




color = 'grey'
plot_params = {'marker':'x',
            'linestyle':'dashed'}

nb_sigma = 1.96

filt = False
timesteps = jnp.arange(T+1)
for model in models: 
    fig, axes = plt.subplots(p_args.state_dim, 1, figsize=(15,15))
    plt.suptitle(model)

    plt.autoscale(True)
    plt.tight_layout()
    smoothed_results, filt_results = eval_model(model)


    if filt: 
        means, stds = filt_results
    else: 
        means, stds = smoothed_results
    for dim in range(p_args.state_dim):
        ax = axes[dim]
        means_d = means[:,dim]
        stds_d = stds[:,dim]

        ax.plot(means_d, label='Pred', color=color, **plot_params)
        ax.fill_between(timesteps, 
                        means_d - nb_sigma*stds_d, 
                        means_d + nb_sigma*stds_d, 
                        alpha=0.2,
                        color=color)
        
        # ax.plot(filt_means[:,dim], label='Filt', **plot_params)
        ax.plot(x[:,dim], label='True', color='black', **plot_params)
        ax.legend()
#%%


