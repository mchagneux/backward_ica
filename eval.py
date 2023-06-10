#%%
import jax, jax.numpy as jnp
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)

import pandas
import matplotlib.pyplot as plt
from src.variational import get_variational_model, NeuralBackwardSmoother
from src.stats.hmm import get_generative_model
from src.utils.misc import *
import os 
path = 'experiments/p_chaotic_rnn/2023_06_10__12_31_12'
num_smoothing_samples = 1000

key = jax.random.PRNGKey(0)
dummy_key = key

p_args = load_args('args', path)
p = get_generative_model(p_args)
theta_star = load_params('theta_star', path)


x = jnp.load(os.path.join(path, 'state_seqs.npy'))[0]
y = jnp.load(os.path.join(path, 'obs_seqs.npy'))[0]
seq_length = len(y)

T = seq_length - 1 
models = ['data/crnn/2023-06-09_14-47-15_Train_run',
          'johnson_backward,200.2.adam,1e-2,cst.reset,500.autodiff_on_backward',
          'johnson_backward,200.2.adam,1e-2,cst.reset,500.score,variance_reduction,bptt_depth_2']

def eval_model(model):
    
    if 'crnn' in model:
        means_filt = jnp.load(os.path.join(model, 'filter_means.npy'))
        covs_filt = jnp.load(os.path.join(model, 'filter_covs.npy'))
        covs_filt = jax.vmap(jnp.diagonal)(covs_filt)
        smoothing_samples = jnp.load(os.path.join(model, 'smoothing_stats.npy'))
        means_smooth = jnp.mean(smoothing_samples, axis=1)
        covs_smooth = jnp.var(smoothing_samples, axis=1)
        return (jnp.array([means_smooth]), jnp.array([covs_smooth])), \
            (jnp.array([means_filt]), jnp.array([covs_filt]))
    else: 
        model_path = os.path.join(path, model)
        q_args = load_args('args', model_path)
        q_args.state_dim, q_args.obs_dim = p_args.state_dim, p_args.obs_dim
        q = get_variational_model(q_args, p)
        
        def eval_from_params(fit_nb):

            phi = load_params(f'phi_fit_{fit_nb}', model_path)

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


            return means_smooth[:seq_length], jnp.sqrt(covs_smooth)[:seq_length], \
                means_filt[:seq_length], jnp.sqrt(covs_filt)[:seq_length]
        means_smooth, covs_smooth, means_filt, covs_filt = [], [], [], []
        for fit_nb in range(q_args.num_fits):
            results = eval_from_params(fit_nb)
            means_smooth.append(results[0])
            covs_smooth.append(results[1])
            means_filt.append(results[2])
            covs_filt.append(results[3])
        return (jnp.array(means_smooth), jnp.array(covs_smooth)), \
                    (jnp.array(means_filt), jnp.array(covs_filt))

y = y[:seq_length]
x = x[:seq_length]

color = 'grey'
plot_params = {'marker':'x',
            'linestyle':'dashed'}

nb_sigma = 1.96

filt = False
timesteps = jnp.arange(seq_length)
for model in models: 
    print(f'Eval of {model}:')

    smoothed_results, filt_results = eval_model(model)


    if filt: 
        means, stds = filt_results
    else: 
        means, stds = smoothed_results

    rmse_avg_over_dims = jnp.mean(jnp.sqrt(jnp.mean((means - x)**2, axis=1)), axis=-1)
    # print('Per-fit RMSE:', rmse_avg_over_dims)

    mean_of_rmses = jnp.mean(rmse_avg_over_dims)
    std_of_rmse = jnp.std(rmse_avg_over_dims)
    print(f'RMSE summary: {mean_of_rmses:.3f} +- {std_of_rmse:.3f}')
    print('-------')


    # fig, axes = plt.subplots(p_args.state_dim, 1, figsize=(15,1.5*p_args.state_dim))

    # plt.autoscale(True)
    # plt.tight_layout()

    # for dim in range(p_args.state_dim):
    #     ax = axes[dim]
    #     means_d = means[:,dim]
    #     stds_d = stds[:,dim]

    #     ax.plot(means_d, label='Pred', color=color, **plot_params)
    #     ax.fill_between(timesteps, 
    #                     means_d - nb_sigma*stds_d, 
    #                     means_d + nb_sigma*stds_d, 
    #                     alpha=0.2,
    #                     color=color)
        
    #     # ax.plot(filt_means[:,dim], label='Filt', **plot_params)
    #     ax.plot(x[:,dim], label='True', color='black', **plot_params)
    #     ax.legend()
    # plt.suptitle(f'model: {model}, rmse: {rmse_avg_over_dims:.3f}')
#%%


