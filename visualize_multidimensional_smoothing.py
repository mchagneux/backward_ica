#%%
from collections import namedtuple
from typing import Dict, NamedTuple
from dataclasses import dataclass, is_dataclass
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
exp_dir = 'experiments/p_nonlinear/p_nonlinear_dim_10_10_stability_tests/trainings/johnson_freeze__theta__transition_phi/2022_07_15__15_21_33'
eval_dir = os.path.join(exp_dir, 'visual_eval')

# shutil.rmtree(eval_dir)
os.makedirs(eval_dir, exist_ok=True)


args = utils.load_args('train_args',exp_dir)

utils.set_global_cov_mode(args)

key_theta = jax.random.PRNGKey(args.seed_theta)
key_phi = jax.random.PRNGKey(args.seed_phi)
num_particles = 1000
num_seqs = 10
seq_length = args.seq_length
p = hmm.NonLinearGaussianHMM(state_dim=args.state_dim, 
                        obs_dim=args.obs_dim, 
                        transition_matrix_conditionning=args.transition_matrix_conditionning,
                        layers=args.emission_map_layers,
                        slope=args.slope,
                        num_particles=num_particles,
                        transition_bias=args.transition_bias,
                        range_transition_map_params=args.range_transition_map_params,
                        injective=args.injective) # specify the structure of the true model
                        
theta_star = utils.load_params('theta', exp_dir)

if 'linear' in args.q_version:

    q = hmm.LinearGaussianHMM(state_dim=args.state_dim, 
                            obs_dim=args.obs_dim,
                            transition_matrix_conditionning=args.transition_matrix_conditionning,
                            transition_bias=args.transition_bias, 
                            emission_bias=False)

elif 'johnson' in args.q_version:
    q = hmm.JohnsonBackwardSmoother(transition_kernel=p.transition_kernel,
                                    obs_dim=args.obs_dim, 
                                    update_layers=args.update_layers,
                                    explicit_proposal='explicit_proposal' in args.q_version)

else:
    q = hmm.GeneralBackwardSmoother(state_dim=args.state_dim, 
                                    obs_dim=args.obs_dim, 
                                    update_layers=args.update_layers,
                                    backwd_layers=args.backwd_map_layers)

phi = utils.load_params('phi', exp_dir)[1]



key_theta, key_gen, key_ffbsi = jax.random.split(key_theta,3)
state_seqs, obs_seqs = p.sample_multiple_sequences(key_gen, theta_star, num_seqs, seq_length)
keys_ffbsi = jax.random.split(key_theta, num_seqs)


# def eval_smoothing_single_seq(state_seq, obs_seq, slices):

#     means_smc = p.smooth_seq_at_multiple_timesteps(key_theta, obs_seq, theta_star, slices)[0]
#     means_q = q.smooth_seq_at_multiple_timesteps(obs_seq, phi, slices)[0]

#     q_vs_states = jnp.mean(jnp.linalg.norm(means_q[-1] - state_seq, ord=1, axis=1), axis=0)
#     ref_vs_states = jnp.mean(jnp.linalg.norm(means_smc[-1] - state_seq, ord=1, axis=1), axis=0)
#     q_vs_ref_marginals = jnp.linalg.norm((means_q[-1] - means_smc[-1]), ord=1, axis=1)[slices]
    
#     q_vs_ref_additive = []
#     for means_smc_n, means_q_n in zip(means_smc, means_q):
#         q_vs_ref_additive.append(jnp.linalg.norm(jnp.sum(means_smc_n - means_q_n, axis=0),ord=1))
#     q_vs_ref_additive = jnp.array(q_vs_ref_additive)

#     return jnp.array([ref_vs_states, q_vs_states, q_vs_ref_additive[-1]]), \
#         q_vs_ref_marginals, \
#         q_vs_ref_additive
       

# eval_smoothing = jax.vmap(eval_smoothing_single_seq, in_axes=(0,0,None))

# num_slices = 10
# slice_length = len(obs_seqs[0]) // num_slices
# slices = jnp.array(list(range(0, len(obs_seqs[0])+1, slice_length)))[1:]

# ref_and_q_vs_states, q_vs_ref_marginals, q_vs_ref_additive = eval_smoothing(state_seqs, obs_seqs, slices)
# ref_and_q_vs_states = pd.DataFrame(ref_and_q_vs_states, columns = ['FFBSi', 'Variational', 'Additive |FFBSi-Variational|'])
# ref_and_q_vs_states.to_csv(os.path.join(eval_dir, 'tabled_results'))


# #%%
# fig, (ax0, ax1) = plt.subplots(2,1, figsize=(30,30))
# plt.tight_layout()
# plt.autoscale(True)

# q_vs_ref_marginals = pd.DataFrame(index=slices, data=q_vs_ref_marginals.T)#.unstack().reset_index(name='value')
# q_vs_ref_additive = pd.DataFrame(index=slices, data=q_vs_ref_additive.T)#.unstack().reset_index(name='value')

# sns.lineplot(ax=ax0, data=q_vs_ref_marginals)#, x='level_1', y='value')
# sns.lineplot(ax=ax1, data=q_vs_ref_additive)#, x='level_1', y='value')

# ax0.set_title('Marginal 1-norm error against SMC')
# ax0.set_xlabel('t')

# ax1.set_title('Additive 1-norm error against SMC')
# ax1.set_xlabel('t')

# plt.savefig(os.path.join(eval_dir, 'marginal_and_additive_errors.pdf'),format='pdf')
# plt.clf()

# filt_dir = os.path.join(eval_dir, 'filt')
# smooth_dir = os.path.join(eval_dir, 'smooth')
# os.makedirs(filt_dir, exist_ok=True)
# os.makedirs(smooth_dir, exist_ok=True)




key_phi, key_filt_q, key_smooth_q = jax.random.split(key_phi, 3)
keys_smooth_q = jax.random.split(key_smooth_q, num_seqs)

print('Bootstrap filtering...')
means_filt_smc, covs_filt_smc = jax.vmap(p.filt_seq_to_mean_cov, in_axes=(0,0,None))(keys_ffbsi, obs_seqs, theta_star)
print('Done.')
print('FFBSi smoothing...')
means_smooth_smc, covs_smooth_smc = jax.vmap(p.smooth_seq_to_mean_cov, in_axes=(0,0,None))(keys_ffbsi, obs_seqs, theta_star)
print('Done.')
if isinstance(q, hmm.GeneralBackwardSmoother) and (not q.backward_help):
    means_filt_q, covs_filt_q = jax.vmap(q.filt_seq, in_axes=(0, None, None))(obs_seqs, phi, num_particles)
    means_smooth_q, covs_smooth_q = jax.vmap(q.smooth_seq, in_axes=(0,0, None, None))(keys_smooth_q, obs_seqs, phi, num_particles)
else:     
    means_filt_q, covs_filt_q = jax.vmap(q.filt_seq, in_axes=(0, None))(obs_seqs, phi)
    means_smooth_q, covs_smooth_q = jax.vmap(q.smooth_seq, in_axes=(0,None))(obs_seqs, phi)

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


#%%
import numpy as np

means_ffbsi = [means_smooth_smc, means_filt_smc]
covs_ffbsi = [covs_smooth_smc, covs_filt_smc]
means_q = [means_smooth_q, means_filt_q]
covs_q = [covs_smooth_q, covs_filt_q]
names = ['smoothing_eval', 'filt_eval']
for mean_ffbsi, cov_ffbsi, mean_q, cov_q, task_name in zip(means_ffbsi, covs_ffbsi, means_q, covs_q, names):

    for seq_nb in range(num_seqs):
        fig, axes = plt.subplots(args.state_dim, 2, figsize=(30,30))
        axes = np.atleast_2d(axes)
        name = f'{task_name}_{seq_nb}'

        for dim_nb in range(args.state_dim):

            utils.plot_relative_errors_1D(axes[dim_nb,0], state_seqs[seq_nb,:,dim_nb], mean_ffbsi[seq_nb,:,dim_nb], cov_ffbsi[seq_nb,:,dim_nb])

        for dim_nb in range(args.state_dim):

            if isinstance(q, hmm.LinearBackwardSmoother) or q.backward_help:
                utils.plot_relative_errors_1D(axes[dim_nb,1], state_seqs[seq_nb,:,dim_nb], mean_q[seq_nb,:,dim_nb], cov_q[seq_nb,:,dim_nb,dim_nb])
            else:
                utils.plot_relative_errors_1D(axes[dim_nb,1], state_seqs[seq_nb,:,dim_nb], mean_q[seq_nb,:,dim_nb], cov_q[seq_nb,:,dim_nb])

        plt.autoscale(True)
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, name+'.pdf'), format='pdf')
        plt.clf()