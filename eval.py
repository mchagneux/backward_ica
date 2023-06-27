#%%
import jax, jax.numpy as jnp
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)


import pandas
import matplotlib.pyplot as plt
from src.variational import get_variational_model, NeuralBackwardSmoother
from src.stats.hmm import get_generative_model, LinearGaussianHMM
from src.utils.misc import *
import os 

path = 'experiments/p_chaotic_rnn/2023_06_27__10_29_35'
models = ['data/crnn/2023-06-26_19-45-59_Train_run',
          'johnson_backward,100.100.adam,1e-3,cst.true_online,50.score,truncated,variance_reduction,paris,bptt_depth_1']
num_smoothing_samples = 1000
plot = True
filt = False
online_fit = True

key = jax.random.PRNGKey(0)
dummy_key = key

p_args = load_args('args', path)
p = get_generative_model(p_args)
theta_star = load_params('theta_star', path)



y = jnp.load(os.path.join(path, 'obs_seqs.npy'))[0]
seq_length = len(y)
T = seq_length - 1

if isinstance(p, LinearGaussianHMM):
    if filt: 
        x = p.filt_seq(y, theta_star)[0]
    else: 
        x = p.smooth_seq(y, theta_star)[0]
else:
    x = jnp.load(os.path.join(path, 'state_seqs.npy'))[0]



def eval_model(model):
    

    if online_fit: 
        if 'crnn' in model: 
            means_tm1_path = os.path.join(model, 'x_Tm1_means.npy')
            means_t_path = os.path.join(model, 'filter_means.npy')
        else: 
            means_tm1_path = os.path.join(path, model, 'x_tm1.npy')
            means_t_path = os.path.join(path, model, 'x_t.npy')
        return jnp.load(means_tm1_path), jnp.load(means_t_path)
    else:      
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

timesteps = jnp.arange(seq_length)

for model in models: 
    print(f'Eval of {model}:')


    if online_fit: 
        means_tm1, means_t = eval_model(model)
        if filt: 
            means = means_t
        else: 
            means = jnp.vstack([means_tm1, means_t[-1]])
            
        rmse_avg_over_dims = jnp.mean(jnp.sqrt(jnp.mean((means - x)**2, axis=1)), axis=-1)
        print(f'RMSE: {rmse_avg_over_dims:.3f}')

        if plot: 

            fig, axes = plt.subplots(p_args.state_dim, 1, figsize=(15,1.5*p_args.state_dim))

            plt.autoscale(True)
            plt.tight_layout()

            for dim in range(p_args.state_dim):
                ax = axes[dim]
                means_d = means[:,dim]

                ax.plot(means_d, label='Pred', color=color, **plot_params)                
                # ax.plot(filt_means[:,dim], label='Filt', **plot_params)
                ax.plot(x[:,dim], label='True', color='black', **plot_params)
                ax.legend()
            # plt.suptitle(f'model: {model}, rmse: {rmse_of_best:.3f}')
            if not 'crnn' in model:
                plt.savefig(os.path.join(path, model, 'eval_best.pdf'), format='pdf')

    else: 

        smoothed_results, filt_results = eval_model(model)




        if filt: 
            means, stds = filt_results
        else: 
            means, stds = smoothed_results


        rmse_avg_over_dims = jnp.mean(jnp.sqrt(jnp.mean((means - x)**2, axis=1)), axis=-1)
        print('Per-fit RMSE:', rmse_avg_over_dims)
        best_fit = jnp.argmin(rmse_avg_over_dims)

        mean_of_rmses = jnp.mean(rmse_avg_over_dims)
        std_of_rmse = jnp.std(rmse_avg_over_dims)

            
        means, stds = means[best_fit], stds[best_fit]
        rmse_of_best = rmse_avg_over_dims[best_fit]
        print(f'RMSE: {mean_of_rmses:.3f} +- {std_of_rmse:.3f} (best {rmse_of_best:.3f})')
        print('-------')
        
        if plot: 


            fig, axes = plt.subplots(p_args.state_dim, 1, figsize=(15,1.5*p_args.state_dim))

            plt.autoscale(True)
            plt.tight_layout()

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
            # plt.suptitle(f'model: {model}, rmse: {rmse_of_best:.3f}')
            plt.savefig(os.path.join(path, model, 'eval_best.pdf'), format='pdf')
#%%


