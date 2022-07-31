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
import pickle 

exp_dir = 'experiments/p_nonlinear/2022_07_30__19_07_19'

method_names = ['johnson_freeze__theta', 
                'johnson_explicit_proposal_freeze__theta']
pretty_names = ['Ours','Johnson']


train_args = utils.load_args('train_args',os.path.join(exp_dir, method_names[0]))
utils.set_global_cov_mode(train_args)

eval_dir = os.path.join(exp_dir, 'eval')
os.makedirs(eval_dir, exist_ok=True)

# shutil.rmtree(eval_dir)




key_theta = jax.random.PRNGKey(train_args.seed_theta)
num_particles = 1000
num_smooth_particles = 1000
num_seqs = 10
seq_length = train_args.seq_length
load = False



p = hmm.NonLinearGaussianHMM(state_dim=train_args.state_dim, 
                        obs_dim=train_args.obs_dim, 
                        transition_matrix_conditionning=train_args.transition_matrix_conditionning,
                        layers=train_args.emission_map_layers,
                        slope=train_args.slope,
                        num_particles=num_particles,
                        num_smooth_particles=num_smooth_particles,
                        transition_bias=train_args.transition_bias,
                        range_transition_map_params=train_args.range_transition_map_params,
                        injective=train_args.injective) # specify the structure of the true model
                        
theta_star = utils.load_params('theta', os.path.join(exp_dir, method_names[0]))
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



keys_ffbsi = jax.random.split(key_theta, num_seqs)
if not load: 
    filt_results, smooth_results = [], []
    print('Bootstrap filtering...')
    means_filt_smc, covs_filt_smc = jax.vmap(p.filt_seq_to_mean_cov, in_axes=(0,0,None))(keys_ffbsi, obs_seqs, theta_star)
    print('Done.')
    print('FFBSi smoothing...')
    means_smooth_smc, covs_smooth_smc = jax.vmap(p.smooth_seq_to_mean_cov, in_axes=(0,0,None))(keys_ffbsi, obs_seqs, theta_star)
    print('Done.')

    filt_results.append((means_filt_smc, covs_filt_smc))
    smooth_results.append((means_smooth_smc, covs_smooth_smc))




qs = []
phis = []
for method_name in method_names:
    method_dir = os.path.join(exp_dir, method_name)
    args = utils.load_args('train_args', method_dir)

    key_phi = jax.random.PRNGKey(args.seed_phi)

    key_phi, key_filt_q, key_smooth_q = jax.random.split(key_phi, 3)
    keys_smooth_q = jax.random.split(key_smooth_q, num_seqs)

    phi = utils.load_params('phi', method_dir)[1]
    phis.append(phi)

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
    qs.append(q)
    if not load: 
        if isinstance(q, hmm.GeneralBackwardSmoother) and (not q.backward_help):
            means_filt_q, covs_filt_q = jax.vmap(q.filt_seq, in_axes=(0, None, None))(obs_seqs, phi, num_particles)
            means_smooth_q, covs_smooth_q = jax.vmap(q.smooth_seq, in_axes=(0,0, None, None))(keys_smooth_q, obs_seqs, phi, num_particles)
        else:     
            means_filt_q, covs_filt_q = jax.vmap(q.filt_seq, in_axes=(0, None))(obs_seqs, phi)
            means_smooth_q, covs_smooth_q = jax.vmap(q.smooth_seq, in_axes=(0,None))(obs_seqs, phi)

        filt_results.append((means_filt_q, covs_filt_q))
        smooth_results.append((means_smooth_q, covs_smooth_q))

if not load: 
    with open(os.path.join(eval_dir, 'sequences.pickle'),'wb') as f:
        pickle.dump((state_seqs, obs_seqs), f)

    with open(os.path.join(eval_dir, 'results.pickle'),'wb') as f:
        pickle.dump((filt_results, smooth_results), f)

#%%
import numpy as np

colors = ['blue',
         'red']
print('Plotting individual sequences...')
for task_name, results in zip(['filtering', 'smoothing'], [filt_results, smooth_results]): 
    for seq_nb in range(num_seqs):
        fig, axes = plt.subplots(train_args.state_dim, len(method_names), sharey=True)
        axes = np.atleast_2d(axes)
        name = f'{task_name}_seq_{seq_nb}'
        for dim_nb in range(train_args.state_dim):
            for method_nb, method_name in enumerate(pretty_names): 
                mean_q, cov_q = results[method_nb+1]
                # axes[dim_nb,method_nb].plot(range(len(state_seqs[seq_nb])), state_seqs[seq_nb,:,dim_nb], ='grey', linestyle='dashed', marker='.', label='True state')
                mean_ffbsi, cov_ffbsi = results[0]
                utils.plot_relative_errors_1D(axes[dim_nb,method_nb], mean_ffbsi[seq_nb,:,dim_nb], cov_ffbsi[seq_nb,:,dim_nb], color='black', alpha=0.2, label='FFBSi', hatch='//')
                # if isinstance(qs[method_nb], hmm.LinearBackwardSmoother) or qs[method_nb].backward_help:
                utils.plot_relative_errors_1D(axes[dim_nb, method_nb], mean_q[seq_nb,:,dim_nb], cov_q[seq_nb,:,dim_nb,dim_nb], color='red', alpha=0.1, label=f'{method_name}')
                # else:
                #     utils.plot_relative_errors_1D(axes[dim_nb, method_nb], mean_q[seq_nb,:,dim_nb], cov_q[seq_nb,:,dim_nb], color=colors[method_nb], alpha=0.1, hatch='/' if method_nb == 0 else None, label=f'{method_name}')
                axes[dim_nb, method_nb].legend()

        plt.autoscale(True)
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, name+'.pdf'), format='pdf')
        plt.clf()


def eval_smoothing_single_seq(state_seq, obs_seq, slices, method_nb):

    means_smc = p.smooth_seq_at_multiple_timesteps(key_theta, obs_seq, theta_star, slices)[0]
    means_q = qs[method_nb].smooth_seq_at_multiple_timesteps(obs_seq, phis[method_nb], slices)[0]

    q_vs_states = jnp.mean(jnp.linalg.norm(means_q[-1] - state_seq, ord=1, axis=1), axis=0)
    ref_vs_states = jnp.mean(jnp.linalg.norm(means_smc[-1] - state_seq, ord=1, axis=1), axis=0)
    q_vs_ref_marginals = jnp.linalg.norm((means_q[-1] - means_smc[-1]), ord=1, axis=1)[slices]
    
    q_vs_ref_additive = []
    for means_smc_n, means_q_n in zip(means_smc, means_q):
        q_vs_ref_additive.append(jnp.linalg.norm(jnp.sum(means_smc_n - means_q_n, axis=0),ord=1))
    q_vs_ref_additive = jnp.array(q_vs_ref_additive)

    return jnp.array([ref_vs_states, q_vs_states, q_vs_ref_additive[-1]]), \
        q_vs_ref_marginals, \
        q_vs_ref_additive
       

eval_smoothing = jax.vmap(eval_smoothing_single_seq, in_axes=(0,0,None,None))

num_slices = 10
slice_length = len(obs_seqs[0]) // num_slices
slices = jnp.array(list(range(0, len(obs_seqs[0])+1, slice_length)))[1:]
# fig, (ax0, ax1) = plt.subplots(2,1)
q_vs_ref_marginals_all = []
q_vs_ref_additive_all = []
ref_and_q_vs_states_all = []

for method_nb, (method_name, pretty_name) in enumerate(zip(method_names, pretty_names)):

    if not load: 
        print(f'Evaluating {method_name}')

        ref_and_q_vs_states, q_vs_ref_marginals, q_vs_ref_additive = eval_smoothing(state_seqs, obs_seqs, slices, method_nb)

        with open(os.path.join(eval_dir, f'eval_{method_name}.pickle'), 'wb') as f:
            pickle.dump((ref_and_q_vs_states, q_vs_ref_marginals, q_vs_ref_additive), f)
        print('Done.')
    else: 
        print(f'Loading {method_name}')

        with open(os.path.join(eval_dir, f'eval_{method_name}.pickle'), 'rb') as f:
            ref_and_q_vs_states, q_vs_ref_marginals, q_vs_ref_additive = pickle.load(f)
        print('Done.')

    ref_and_q_vs_states = pd.DataFrame(ref_and_q_vs_states, columns = ['FFBSi', f'{pretty_name}', f'Additive |FFBSi-{pretty_name}|'])
    ref_and_q_vs_states_all.append(ref_and_q_vs_states)
    ref_and_q_vs_states.to_csv(os.path.join(eval_dir, 'tabled_results.csv'))

    q_vs_ref_marginals = pd.DataFrame(index=slices, data=q_vs_ref_marginals.T).unstack().reset_index(name='value')
    q_vs_ref_additive = pd.DataFrame(index=slices, data=q_vs_ref_additive.T).unstack().reset_index(name='value')

    q_vs_ref_marginals_all.append(q_vs_ref_marginals)
    q_vs_ref_additive_all.append(q_vs_ref_additive)


ref_and_q_vs_states = pd.concat(ref_and_q_vs_states_all, axis=1)
ref_and_q_vs_states = ref_and_q_vs_states.T.drop_duplicates().T
ref_and_q_vs_states = ref_and_q_vs_states[['FFBSi','Ours','Johnson','Additive |FFBSi-Ours|','Additive |FFBSi-Johnson|']]
print(ref_and_q_vs_states.to_latex(float_format="%.2f" ))
q_vs_ref_marginals = pd.concat(q_vs_ref_marginals_all, keys=pretty_names)
q_vs_ref_marginals.columns = ['Sequence', 'Timestep', 'Value']
q_vs_ref_marginals = q_vs_ref_marginals.reset_index(level=0).reset_index(drop=True)
q_vs_ref_marginals.columns = ['Method', 'Sequence', 'Timestep', 'Value']

q_vs_ref_additive = pd.concat(q_vs_ref_additive_all, keys=pretty_names)
q_vs_ref_additive.columns = ['Sequence', 'Timestep', 'Value']
q_vs_ref_additive = q_vs_ref_additive.reset_index(level=0).reset_index(drop=True)
q_vs_ref_additive.columns = ['Method', 'Sequence', 'Timestep', 'Value']


import numpy as np
sns.lineplot(data=q_vs_ref_marginals, x='Timestep', y='Value', hue='Method', style='Sequence')
# plt.title('Marginal 1-norm error against FFBSi')
plt.savefig(os.path.join(eval_dir, f'marginal_errors.pdf'),format='pdf')
plt.clf()
sns.lineplot(data=q_vs_ref_additive, x='Timestep', y='Value',hue='Method', style='Sequence')
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


