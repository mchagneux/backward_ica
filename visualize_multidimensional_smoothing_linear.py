#%%
from typing import NamedTuple
import haiku as hk 
import jax 
import jax.numpy as jnp
import backward_ica.hmm as hmm 
import backward_ica.utils as utils 
import backward_ica.smc as smc
import seaborn as sns
import os 
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import pandas as pd
from pandas.plotting import table
import math
import pickle 
from backward_ica.elbos import BackwardLinearELBO


exp_dir = 'experiments/p_linear/2022_08_02__15_26_01'



     
train_args = utils.load_args('train_args',os.path.join(exp_dir))
utils.set_defaults(train_args)

eval_dir = os.path.join(exp_dir, 'eval')
os.makedirs(eval_dir, exist_ok=True)
from time import time

# shutil.rmtree(eval_dir)


def check_kl_divergences(state_seq, obs_seq, p:hmm.LinearGaussianHMM, q:hmm.LinearGaussianHMM, theta:utils.HMMParams, phi:utils.HMMParams, h = lambda x:x):
    filt_seq = q.filt_seq(obs_seq, phi)
    formatted_phi = q.format_params(phi)
    backwd_seq = q.compute_backwd_params_seq(q.compute_filt_params_seq(obs_seq, formatted_phi), formatted_phi)

    def marginalize(mean, cov, linear_kernel_params):
        A = linear_kernel_params.map.w
        b = linear_kernel_params.map.b
        Q = linear_kernel_params.scale.cov
        new_mean = A @ mean + b 
        new_cov = A @ cov @ A.T + Q
        return new_mean, new_cov 
    right_members = jax.vmap(marginalize, in_axes=(0,0,None))(filt_seq[0][:-1], filt_seq[1][:-1], p.format_params(theta).transition)
    left_members = jax.vmap(marginalize, in_axes=(0,0,0))(filt_seq[0][1:], filt_seq[1][1:], backwd_seq)

    def left_norm(left_gaussian, right_gaussian):
        return jnp.linalg.norm(left_gaussian[0] - right_gaussian[0], ord=1)
        
        
    def kl_gaussian(left_gaussian, right_gaussian):
        mean_p, cov_p = left_gaussian
        mean_q, cov_q = right_gaussian 
        inv_cov_q = jnp.linalg.inv(cov_q)
        return 0.5 * (jnp.log(jnp.linalg.det(cov_q) / jnp.linalg.det(cov_p)) \
                    - cov_p.shape[0] \
                    + jnp.trace(inv_cov_q @ cov_p) \
                    + (mean_p - mean_q).T @ inv_cov_q @ (mean_p - mean_q))

    inf_norm_h = jnp.linalg.norm(jnp.linalg.norm(jax.vmap(h)(state_seq), axis=1), ord='inf')

    upper_bounds_tv = jnp.sqrt(jax.vmap(kl_gaussian)(left_members, right_members) / 2) 
    left_terms = jax.vmap(left_norm)(left_members, right_members) 

    return left_terms, 2 * upper_bounds_tv * inf_norm_h



key_theta = jax.random.PRNGKey(train_args.seed_theta)
num_seqs = 50
seq_length = 500
load = True
metrics = True
plot_sequences = False
recompute_marginals = False
profile = False
check_divergences = True




def profile_q(key, p, q, theta, phi, obs_seqs):


    inference_f = jax.jit(jax.vmap(q.smooth_seq, in_axes=(0, None)))
    inference_f(obs_seqs[:20], phi)[0].block_until_ready()
    time0 = time()
    inference_f(obs_seqs, phi)[0].block_until_ready()
    time_inference =  (time() - time0) / len(obs_seqs)

    elbo =  jax.jit(lambda keys, obs_seqs, theta, phi: jax.vmap(BackwardLinearELBO(p,q,1), in_axes=(0,0,None,None))(keys, obs_seqs, p.format_params(theta), q.format_params(phi)))
    elbo_grad = jax.jit(jax.vmap(jax.grad(lambda key, obs_seq, theta, phi: BackwardLinearELBO(p,q,1)(key, obs_seq, p.format_params(theta), q.format_params(phi)), argnums=3), in_axes=(0,0,None,None)))
    keys = jax.random.split(key, len(obs_seqs))
    elbo(keys[:20], obs_seqs[:20], theta, phi).block_until_ready()
    print(elbo_grad(keys[:20], obs_seqs[:20], theta, phi))

    time0 = time()
    elbo(keys, obs_seqs, theta, phi).block_until_ready()
    time_elbo = (time() - time0) / len(obs_seqs)

    time0 = time()
    print(elbo_grad(keys, obs_seqs, theta, phi))
    time_grad_elbo = (time() - time0) / len(obs_seqs)

    return time_inference, time_elbo, time_grad_elbo

    


p = hmm.LinearGaussianHMM(state_dim=train_args.state_dim, 
                        obs_dim=train_args.obs_dim, 
                        transition_matrix_conditionning=train_args.transition_matrix_conditionning,
                        range_transition_map_params=train_args.range_transition_map_params,
                        transition_bias=train_args.transition_bias,
                        emission_bias=train_args.emission_bias) 
                        
theta_star = utils.load_params('theta', os.path.join(exp_dir))

if load: 
    print('Loading sequences and results...')
    with open(os.path.join(eval_dir, 'sequences.pickle'),'rb') as f:
        state_seqs, obs_seqs = pickle.load(f)
    with open(os.path.join(eval_dir, 'results.pickle'),'rb') as f:
        filt_results, smooth_results = pickle.load(f)
    print('Done.')
else:
    key_theta, key_gen, key_ffbsi = jax.random.split(key_theta,3)
    state_seqs, obs_seqs = p.sample_multiple_sequences(key_gen, theta_star, num_seqs, seq_length)



if not load: 
    filt_results, smooth_results = [], []
    print('Kalman filtering...')
    means_filt_kalman, covs_filt_kalman = jax.vmap(p.filt_seq, in_axes=(0, None))(obs_seqs, theta_star)
    print('Done.')
    print('Kalman smoothing...')
    means_smooth_kalman, covs_smooth_kalman = jax.vmap(p.smooth_seq, in_axes=(0, None))(obs_seqs, theta_star)

    print('Done.')

    filt_results.append((means_filt_kalman, covs_filt_kalman))
    smooth_results.append((means_smooth_kalman, covs_smooth_kalman))

phis = utils.load_params('phi', exp_dir)
q = hmm.LinearGaussianHMM(train_args.state_dim, 
            train_args.obs_dim, 
            'diagonal',
            range_transition_map_params=(0,1),
            transition_bias=train_args.transition_bias,
            emission_bias=train_args.emission_bias)
test = 0 


epoch_nbs = [110, 140, 170]
pretty_names = [f'Epoch {epoch_nb}' for epoch_nb in epoch_nbs]

qs = []
current_palette = sns.color_palette()
def plot_training_curve(exp_dir, epoch_nbs):
    with open(os.path.join(exp_dir, 'train_logs'), 'rb') as f:
        start_pt = 60
        avg_evidence, avg_elbos = pickle.load(f)
        plt.axhline(y=avg_evidence, c='black', linestyle='dotted')
        for idx, epoch_nb in enumerate(epoch_nbs):
            plt.axvline(x=epoch_nb, label=f'Epoch {epoch_nb}', c=current_palette[idx])
        plt.plot(range(start_pt, len(avg_elbos)), avg_elbos[start_pt:], c='black')
        plt.xlabel('Epoch')
        plt.legend()
        plt.tight_layout()
        plt.autoscale(True)
        plt.savefig(os.path.join(eval_dir, 'training_curves.pdf'), format='pdf')
        plt.clf()

plot_training_curve(exp_dir, epoch_nbs)



for epoch_nb in epoch_nbs: 



    phi = phis[epoch_nb][1]

#     if profile: 
#         state_seqs_profile, obs_seqs_profile = p.sample_multiple_sequences(key_theta, theta_star, 1000, 100)
#         print(f'Computational time {method_name}', profile_q(key_phi, p, q, theta_star, phi, obs_seqs_profile))

#     qs.append(q)
    if not load: 
        means_filt_q, covs_filt_q = jax.vmap(q.filt_seq, in_axes=(0, None))(obs_seqs, phi)
        means_smooth_q, covs_smooth_q = jax.vmap(q.smooth_seq, in_axes=(0,None))(obs_seqs, phi)

        filt_results.append((means_filt_q, covs_filt_q))
        smooth_results.append((means_smooth_q, covs_smooth_q))

    if check_divergences: 

        left_members, right_members = jax.vmap(check_kl_divergences, in_axes=(0,0, None, None, None, None))(state_seqs, obs_seqs, p, q, theta_star, phi)
        df_dict = {}
        print(left_members.shape)
        print(right_members.shape)



if not load: 
    with open(os.path.join(eval_dir, 'sequences.pickle'),'wb') as f:
        pickle.dump((state_seqs, obs_seqs), f)

    with open(os.path.join(eval_dir, 'results.pickle'),'wb') as f:
        pickle.dump((filt_results, smooth_results), f)

#%%
import numpy as np

if plot_sequences: 
    colors = ['blue',
            'red']
    print('Plotting individual sequences...')
    for task_name, results in zip(['filtering', 'smoothing'], [filt_results, smooth_results]): 
        for seq_nb in range(num_seqs):
            fig, axes = plt.subplots(train_args.state_dim, 2, sharey=True, figsize=(30,20))
            axes = np.atleast_2d(axes)
            name = f'{task_name}_seq_{seq_nb}'
            for dim_nb in range(train_args.state_dim):
                axes[dim_nb,0].plot(range(len(state_seqs[seq_nb])), state_seqs[seq_nb,:,dim_nb], color='grey', linestyle='dashed', marker='.', label='True state')
                axes[dim_nb,1].plot(range(len(state_seqs[seq_nb])), state_seqs[seq_nb,:,dim_nb], color='grey', linestyle='dashed', marker='.', label='True state')

                mean_kalman, cov_kalman = results[0]
                utils.plot_relative_errors_1D(axes[dim_nb,0], mean_kalman[seq_nb,:,dim_nb], cov_kalman[seq_nb,:,dim_nb,dim_nb], color='black', alpha=0.1, label='Kalman with $\\theta^{*}$', hatch='//')

                mean_q, cov_q = results[-1]
                # if isinstance(qs[method_nb], hmm.LinearBackwardSmoother) or qs[method_nb].backward_help:
                utils.plot_relative_errors_1D(axes[dim_nb,1], mean_q[seq_nb,:,dim_nb], cov_q[seq_nb,:,dim_nb,dim_nb], color='red', alpha=0.2, label=f'Variational {pretty_names[-1]}')
                # else:
                #     utils.plot_relative_errors_1D(axes[dim_nb, method_nb], mean_q[seq_nb,:,dim_nb], cov_q[seq_nb,:,dim_nb], color=colors[method_nb], alpha=0.1, hatch='/' if method_nb == 0 else None, label=f'{method_name}')
                axes[dim_nb,0].legend()
                axes[dim_nb,1].legend()


            plt.autoscale(True)
            plt.tight_layout()
            plt.savefig(os.path.join(eval_dir, name+'.pdf'), format='pdf')
            plt.clf()


def eval_smoothing_single_seq(state_seq, obs_seq, slices, epoch_nb):

    means_kalman = p.smooth_seq_at_multiple_timesteps(obs_seq, theta_star, slices)[0]
    means_q = q.smooth_seq_at_multiple_timesteps(obs_seq, phis[epoch_nb][1], slices)[0]

    q_vs_states = jnp.mean(jnp.linalg.norm(means_q[-1] - state_seq, ord=1, axis=1), axis=0)
    ref_vs_states = jnp.mean(jnp.linalg.norm(means_kalman[-1] - state_seq, ord=1, axis=1), axis=0)
    q_vs_ref_marginals = jnp.linalg.norm((means_q[-1] - means_kalman[-1]), ord=1, axis=1)
    
    q_vs_ref_additive = []
    for means_kalman_n, means_q_n in zip(means_kalman, means_q):
        q_vs_ref_additive.append(jnp.linalg.norm(jnp.sum(means_kalman_n - means_q_n, axis=0),ord=1))
    q_vs_ref_additive = jnp.array(q_vs_ref_additive)

    return jnp.array([ref_vs_states, q_vs_states, q_vs_ref_additive[-1]]), \
                    q_vs_ref_marginals, \
                    q_vs_ref_additive
       

eval_smoothing = jax.vmap(eval_smoothing_single_seq, in_axes=(0,0,None,None))


# def recompute_marginals_func(results, method_nb):
#     means_smc = results[0][0]
#     means_q = results[method_nb+1][0]
#     def compute_marginal(means_smc, means_q):
#         return jnp.linalg.norm(means_q - means_smc, ord=1, axis=1)
#     return jax.vmap(compute_marginal)(means_smc, means_q)

# def compute_ffbsi_stds(smooth_results, state_seqs):
#     def compute_ref_vs_states(means_smc, state_seq):
#         ref_vs_states = jnp.linalg.norm(means_smc[-1] - state_seq, ord=1, axis=1)
#         return jnp.var(ref_vs_states, axis=0)

#     return jax.vmap(compute_ref_vs_states)(smooth_results[0][0], state_seqs)

# def compute_mae_marginals(results, method_nb):
#     means_smc = results[0][0]
#     means_q = results[method_nb+1][0]
#     def compute_marginal_mae(means_smc, means_q):
#         return jnp.mean(jnp.linalg.norm(means_q - means_smc, ord=1, axis=1), axis=0)
#     return jax.vmap(compute_marginal_mae)(means_smc, means_q)

if metrics: 

    num_slices = 20
    slice_length = len(obs_seqs[0]) // num_slices
    slices = jnp.array(list(range(0, len(obs_seqs[0])+1, slice_length)))[1:]
    # fig, (ax0, ax1) = plt.subplots(2,1)
    q_vs_ref_marginals_all = []
    q_vs_ref_additive_all = []
    ref_and_q_vs_states_all = []

    for epoch_nb, pretty_name in zip(epoch_nbs, pretty_names):

        if not load: 
            print(f'Evaluating epoch {epoch_nb}')

            ref_and_q_vs_states, q_vs_ref_marginals, q_vs_ref_additive = eval_smoothing(state_seqs, obs_seqs, slices, epoch_nb)

            with open(os.path.join(eval_dir, f'eval_{epoch_nb}.pickle'), 'wb') as f:
                pickle.dump((ref_and_q_vs_states, q_vs_ref_marginals, q_vs_ref_additive), f)
            print('Done.')
        else: 
            print(f'Loading {epoch_nb}')

            with open(os.path.join(eval_dir, f'eval_{epoch_nb}.pickle'), 'rb') as f:
                ref_and_q_vs_states, q_vs_ref_marginals, q_vs_ref_additive = pickle.load(f)
            print('Done.')

        ref_and_q_vs_states = pd.DataFrame(ref_and_q_vs_states, columns = ['FFBSi', f'{pretty_name}', f'Additive |FFBSi-{pretty_name}|'])
        ref_and_q_vs_states_all.append(ref_and_q_vs_states)

        q_vs_ref_marginals = pd.DataFrame(data=q_vs_ref_marginals.T / train_args.state_dim).unstack().reset_index(name='value')
        q_vs_ref_additive = pd.DataFrame(index=slices, data=q_vs_ref_additive.T / train_args.state_dim).unstack().reset_index(name='value')

        q_vs_ref_marginals_all.append(q_vs_ref_marginals)
        q_vs_ref_additive_all.append(q_vs_ref_additive)


    ref_and_q_vs_states = pd.concat(ref_and_q_vs_states_all, axis=1)
    ref_and_q_vs_states = ref_and_q_vs_states.T.drop_duplicates().T

    end_table = ref_and_q_vs_states[[*(f'Additive |FFBSi-{pretty_name}|' for pretty_name in pretty_names)]] / train_args.state_dim

    print(end_table.to_latex(float_format="%.2f" ))
    q_vs_ref_marginals = pd.concat(q_vs_ref_marginals_all, keys=pretty_names)
    q_vs_ref_marginals.columns = ['Sequence', 'n', 'Value']
    q_vs_ref_marginals = q_vs_ref_marginals.reset_index(level=0).reset_index(drop=True)
    q_vs_ref_marginals.columns = ['Stopping time', 'Sequence', 'n', 'Value']

    q_vs_ref_additive = pd.concat(q_vs_ref_additive_all, keys=pretty_names)
    q_vs_ref_additive.columns = ['Sequence', 'n', 'Value']
    q_vs_ref_additive = q_vs_ref_additive.reset_index(level=0).reset_index(drop=True) 
    q_vs_ref_additive.columns = ['Stopping time', 'Sequence', 'n', 'Value']


    import numpy as np
    sns.lineplot(data=q_vs_ref_marginals, x='n', y='Value', hue='Stopping time')
    # plt.title('Marginal 1-norm error against FFBSi')
    plt.ylabel('Marginal error')

    plt.autoscale(True)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, f'marginal_errors.pdf'),format='pdf')

    plt.clf()

    sns.lineplot(data=q_vs_ref_additive, x='n', y='Value', hue='Stopping time')
    sns.lineplot(data=q_vs_ref_additive, x='n', y='Value',hue='Stopping time',style='Sequence',alpha=0.1, legend=False)
    plt.ylabel('Additive error')
    plt.autoscale(True)
    plt.tight_layout()

    # sns.lineplot(data=q_vs_ref_additive, x='Timestep', y='Value',hue='Method', ax=ax)
    # plt.title('Additive 1-norm error against FFBSi')

    plt.savefig(os.path.join(eval_dir, f'additive_errors.pdf'),format='pdf')
    plt.clf()

# ax0.set_title('Marginal 1-norm error against FFBSi')
# ax0.set_xlabel('t')

# ax1.set_title('Additive 1-norm error against FFBSi')
# ax1.set_xlabel('t')



#%%
# bins=100
# for timestep in range(len(filt_weights)):
#     smoothing_particles_t = smoothing_paths[timestep]
#     filt_weights_t = filt_weights[timestep]
#     filt_particles_t = filt_particles[timestep]

#     if args.state_dim == 2:
#         g = sns.JointGrid(x=filt_particles_t[:,0], y=filt_particles_t[:,1], xlim=(-1.5, 1.5), ylim=(-1.5,1.5))

#         g.plot_marginals(sns.histplot, weights=filt_weights_t, bins=bins, binrange=(-1.5, 1.5))
#         g.plot_joint(sns.kdeplot, weights=filt_weights_t, fill=True)

#         g.savefig(os.path.join(filt_dir, str(timestep)))

#         g = sns.JointGrid(x=smoothing_particles_t[:,0], y=smoothing_particles_t[:,1])
#         g.plot_joint(sns.kdeplot, fill=True)
#         g.plot_marginals(sns.histplot, bins=bins, binrange=(-1.5, 1.5))
#         g.savefig(os.path.join(smooth_dir, str(timestep)))

#     elif args.state_dim == 1: 
#         g = sns.displot(x=filt_particles_t[:,0], kde=True, weights = filt_weights_t, binrange=(-1.5, 1.5), bins=bins)
#         g.savefig(os.path.join(filt_dir, str(timestep)))

#         g = sns.displot(smoothing_particles_t, kde=True, bins=bins, binrange=(-1.5, 1.5))
#         g.savefig(os.path.join(smooth_dir, str(timestep)))

#     else: 
#         pass


