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
import pickle

utils.enable_x64(True)

exp_dir = 'experiments/p_chaotic_rnn/2022_10_10__16_40_43'

method_names = ['neural_backward_linear', 
                'external_campbell']
                
pretty_names = ['Ours', 
                'Campbell']

train_args = utils.load_args('train_args', os.path.join(exp_dir, method_names[0]))
if method_names[1] == 'external_campbell':

    train_args.loaded_data = (os.path.join(utils.chaotic_rnn_base_dir, 'x_data.npy'), 
                            os.path.join(utils.chaotic_rnn_base_dir,'y_data.npy'))
    train_args.num_seqs = 1
    train_args.seq_length = 500
    
utils.set_parametrization(train_args)

eval_dir = os.path.join(exp_dir, 'eval')
os.makedirs(eval_dir, exist_ok=True)
from time import time

# shutil.rmtree(eval_dir)

key_theta = jax.random.PRNGKey(train_args.seed_theta)
num_particles = 1000
num_smooth_particles = 1000
num_seqs = 1
seq_length = train_args.seq_length
load = False
metrics = True
plot_sequences = True
recompute_marginals = False
profile = False
filter_rmse = True
visualize_init = False
lag = None
ref_type = 'states'


train_args.num_particles = num_particles
train_args.num_smooth_particles = num_smooth_particles

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

    
p = utils.get_generative_model(train_args)
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
    if train_args.loaded_data: 
        state_seqs, obs_seqs = p.sample_multiple_sequences(key_gen, theta_star, train_args.num_seqs, train_args.seq_length, train_args.single_split_seq, train_args.loaded_data)
    else:
        state_seqs, obs_seqs = p.sample_multiple_sequences(key_gen, theta_star, num_seqs, seq_length)

    # print(state_seqs.shape)
    # print(state_seqs.dtype)

keys_ffbsi = jax.random.split(key_theta, num_seqs)
if not load: 
    filt_results, smooth_results = [], []

    if ref_type == 'smc':
        print('Bootstrap filtering...')

        means_filt_smc, covs_filt_smc = jax.vmap(p.filt_seq_to_mean_cov, in_axes=(0,0,None))(keys_ffbsi, obs_seqs, theta_star)

        print('Done.')
        print('FFBSi smoothing...')
        means_smooth_smc, covs_smooth_smc = jax.vmap(p.smooth_seq_to_mean_cov, in_axes=(0,0,None))(keys_ffbsi, obs_seqs, theta_star)
        print('Done.')

        filt_results.append((means_filt_smc, covs_filt_smc))
        smooth_results.append((means_smooth_smc, covs_smooth_smc))
    else: 
        filt_results.append(None)
        smooth_results.append(None)


class ExternalVariationalFamily():

    def __init__(self, save_dir):

        self.means_filt_q = jnp.load(os.path.join(save_dir, 'filter_means.npy'))[jnp.newaxis,:]
        self.covs_filt_q = jnp.load(os.path.join(save_dir, 'filter_covs.npy'))[jnp.newaxis,:]
        with open(os.path.join(save_dir, 'smoothed_stats.pickle'), 'rb') as f: 
            smoothed_means, smoothed_covs = pickle.load(f)
        self.means_smooth_q_list = smoothed_means
        self.covs_smooth_q_list = smoothed_covs

    def get_filt_means_and_covs(self):
        return (self.means_filt_q, self.covs_filt_q)
    
    def get_smooth_means_and_covs(self):
        return (self.means_smooth_q_list[-1][jnp.newaxis,:], self.covs_smooth_q_list[-1][jnp.newaxis,:])

    def smooth_seq_at_multiple_timesteps(self, obs_seq, phi, slices):

        smoothed_means = [self.means_smooth_q_list[timestep-1] for timestep in slices]
        smoothed_covs = [self.covs_smooth_q_list[timestep-1] for timestep in slices]

        return (smoothed_means, smoothed_covs)


qs = []
phis = []
for method_name in method_names:
    if 'external' in method_name:
        q = ExternalVariationalFamily(utils.chaotic_rnn_base_dir)

        filt_results.append(q.get_filt_means_and_covs())
        smooth_results.append(q.get_smooth_means_and_covs())

        phi = None
    else: 
        method_dir = os.path.join(exp_dir, method_name)
        args = utils.load_args('train_args', method_dir)

        key_phi = jax.random.PRNGKey(args.seed_phi)

        key_phi, key_filt_q, key_smooth_q = jax.random.split(key_phi, 3)
        keys_smooth_q = jax.random.split(key_smooth_q, num_seqs)

        q = utils.get_variational_model(args, p)

        if visualize_init: 
            phi = q.get_random_params(key_phi, args)
        else:
            phi = utils.load_params('phi', method_dir)

        if profile: 
            state_seqs_profile, obs_seqs_profile = p.sample_multiple_sequences(key_theta, theta_star, 1000, 100)
            print(f'Computational time {method_name}', profile_q(key_phi, p, q, theta_star, phi, obs_seqs_profile))

        if not load: 
            if isinstance(q, hmm.NeuralBackwardSmoother) and (not q.backward_help):
                means_filt_q, covs_filt_q = jax.vmap(q.filt_seq, in_axes=(0, None))(obs_seqs, phi)
                means_smooth_q, covs_smooth_q = jax.vmap(q.smooth_seq, in_axes=(0,0, None, None))(keys_smooth_q, obs_seqs, phi, num_particles)
            else:     
                means_filt_q, covs_filt_q = jax.vmap(q.filt_seq, in_axes=(0, None))(obs_seqs, phi)
                means_smooth_q, covs_smooth_q = jax.vmap(q.smooth_seq, in_axes=(0,None,None))(obs_seqs, phi, lag)

            filt_results.append((means_filt_q, covs_filt_q))
            smooth_results.append((means_smooth_q, covs_smooth_q))

    qs.append(q)
    phis.append(phi)

if not load: 
    with open(os.path.join(eval_dir, 'sequences.pickle'),'wb') as f:
        pickle.dump((state_seqs, obs_seqs), f)

    with open(os.path.join(eval_dir, 'results.pickle'),'wb') as f:
        pickle.dump((filt_results, smooth_results), f)





if filter_rmse: 
    if ref_type == 'smc':
        filt_rmse_smc = jnp.mean(jnp.sqrt(jnp.mean((filt_results[0][0] - state_seqs)**2, axis=-1)))
        print('Filter RMSE SMC:', filt_rmse_smc)
        smooth_rmse_smc = jnp.mean(jnp.sqrt(jnp.mean((smooth_results[0][0] - state_seqs)**2, axis=-1)))
        print('Smooth RMSE SMC:', smooth_rmse_smc)
    for method_nb, (method_name, pretty_name) in enumerate(zip(method_names, pretty_names)): 
        filt_rmse_q = jnp.mean(jnp.sqrt(jnp.mean((filt_results[method_nb+1][0] - state_seqs)**2, axis=-1)))
        print(f'Filter RMSE {pretty_name}:', filt_rmse_q)
        smooth_rmse_q = jnp.mean(jnp.sqrt(jnp.mean((smooth_results[method_nb+1][0] - state_seqs)**2, axis=-1)))
        print(f'Smoothing RMSE {pretty_name}:', smooth_rmse_q)
        print('-----')
    if method_name == 'campbell':
        filt_rmses_campbell = jnp.load(os.path.join(utils.chaotic_rnn_base_dir, 'filter_RMSEs.npy'))[:,-1]
        print(f'Filter RMSE campbell external:', jnp.mean(filt_rmses_campbell))

#%%
import numpy as np

if plot_sequences: 
    colors = ['blue',
            'red']
    print('Plotting individual sequences...')
    for task_name, results in zip(['filtering','smoothing'], [filt_results, smooth_results]): 
        for seq_nb in range(num_seqs):
            fig, axes = plt.subplots(train_args.state_dim, len(method_names), sharey='row', figsize=(30,30))
            plt.autoscale(True)
            plt.tight_layout()
            if len(method_names) > 1: axes = np.atleast_2d(axes)
            name = f'{task_name}_seq_{seq_nb}'
            for dim_nb in range(train_args.state_dim):
                for method_nb, method_name in enumerate(pretty_names): 
                    if ref_type == 'smc':
                        mean_ffbsi, cov_ffbsi = results[0]
                        utils.plot_relative_errors_1D(axes[dim_nb,method_nb], mean_ffbsi[seq_nb,:,dim_nb], cov_ffbsi[seq_nb,:,dim_nb], color='black', alpha=0.1, label='FFBSi', hatch='//')
                    axes[dim_nb,method_nb].plot(range(len(state_seqs[seq_nb])), state_seqs[seq_nb,:,dim_nb], color='green', linestyle='dashed', label='True state')
                    mean_q, cov_q = results[method_nb+1]
                    utils.plot_relative_errors_1D(axes[dim_nb, method_nb], mean_q[seq_nb,:,dim_nb], cov_q[seq_nb,:,dim_nb,dim_nb], color='red', alpha=0.2, label=f'{method_name}')
                    axes[dim_nb, method_nb].legend()
            plt.savefig(os.path.join(eval_dir, name))
            plt.close()


def recompute_marginals_func(results, method_nb):
    means_smc = results[0][0]
    means_q = results[method_nb+1][0]
    def compute_marginal(means_smc, means_q):
        return jnp.linalg.norm(means_q - means_smc, ord=1, axis=1)
    return jax.vmap(compute_marginal)(means_smc, means_q)

def compute_ffbsi_stds(means_smc, state_seqs):
    def compute_ref_vs_states(means_smc, state_seq):
        ref_vs_states = jnp.linalg.norm(means_smc[-1] - state_seq, ord=1, axis=1)
        return jnp.var(ref_vs_states, axis=0)

    return jax.vmap(compute_ref_vs_states)(means_smc, state_seqs)

def compute_mae_marginals(means_ref, results, method_nb):
    means_q = results[method_nb+1][0]
    def compute_marginal_mae(means_smc, means_q):
        return jnp.mean(jnp.linalg.norm(means_q - means_smc, ord=1, axis=1), axis=0)
    return jax.vmap(compute_marginal_mae)(means_ref, means_q)


def eval_smoothing_single_seq(state_seq, obs_seq, means_ref, slices, method_nb):


    means_q = qs[method_nb].smooth_seq_at_multiple_timesteps(obs_seq, phis[method_nb], slices)[0]
    # if method_nb == 0: 
    #     means_q[-1] = smooth_results[1][0][0]
    q_vs_states = jnp.mean(jnp.linalg.norm(means_q[-1] - state_seq, ord=1, axis=1), axis=0)
    ref_vs_states = jnp.mean(jnp.linalg.norm(means_ref[-1] - state_seq, ord=1, axis=1), axis=0)
    q_vs_ref_marginals = jnp.linalg.norm((means_q[-1] - means_ref[-1]), ord=1, axis=1)[slices]
    
    q_vs_ref_additive = []
    for means_ref_n, means_q_n in zip(means_ref, means_q):
        q_vs_ref_additive.append(jnp.linalg.norm(jnp.sum(means_ref_n - means_q_n, axis=0),ord=1))
    q_vs_ref_additive = jnp.array(q_vs_ref_additive)

    return jnp.array([ref_vs_states, q_vs_states, q_vs_ref_additive[-1]]), \
                    q_vs_ref_marginals, \
                    q_vs_ref_additive
       

eval_smoothing = jax.vmap(eval_smoothing_single_seq, in_axes=(0,0,0, None,None))

if metrics: 

    num_slices = 250
    slice_length = len(obs_seqs[0]) // num_slices
    slices = jnp.array(list(range(0, len(obs_seqs[0])+1, slice_length)))[1:]
    q_vs_ref_marginals_all = []
    q_vs_ref_additive_all = []
    ref_and_q_vs_states_all = []
    if ref_type == 'smc':
        print('Computing SMC smoothing at multiple timesteps...')
        means_ref = jax.vmap(p.smooth_seq_at_multiple_timesteps, in_axes=(None, 0, None, None))(key_theta, obs_seqs, theta_star, slices)[0]
    elif ref_type == 'states':
        means_ref = [state_seqs[:,:timestep] for timestep in slices]

    
    for method_nb, (method_name, pretty_name) in enumerate(zip(method_names, pretty_names)):

        if not load: 
            print(f'Evaluating {method_name}')

            ref_and_q_vs_states, q_vs_ref_marginals, q_vs_ref_additive = eval_smoothing(state_seqs, obs_seqs, means_ref, slices, method_nb)

            with open(os.path.join(eval_dir, f'eval_{method_name}.pickle'), 'wb') as f:
                pickle.dump((ref_and_q_vs_states, q_vs_ref_marginals, q_vs_ref_additive), f)
            print('Done.')
        else: 
            print(f'Loading {method_name}')

            with open(os.path.join(eval_dir, f'eval_{method_name}.pickle'), 'rb') as f:
                ref_and_q_vs_states, q_vs_ref_marginals, q_vs_ref_additive = pickle.load(f)
            print('Done.')
            if recompute_marginals: 
                print('Recomputing marginals...')
                q_vs_ref_marginals = recompute_marginals_func(smooth_results, method_nb)

        ref_and_q_vs_states = pd.DataFrame(ref_and_q_vs_states, columns = ['Ref.', f'{pretty_name}', f'Additive |Ref.-{pretty_name}|'])
        ref_and_q_vs_states_all.append(ref_and_q_vs_states)

        q_vs_ref_marginals = pd.DataFrame(data=q_vs_ref_marginals.T / train_args.state_dim).unstack().reset_index(name='value')
        q_vs_ref_additive = pd.DataFrame(index=slices, data=q_vs_ref_additive.T / train_args.state_dim).unstack().reset_index(name='value')

        q_vs_ref_marginals_all.append(q_vs_ref_marginals)
        q_vs_ref_additive_all.append(q_vs_ref_additive)


    ref_and_q_vs_states = pd.concat(ref_and_q_vs_states_all, axis=1)
    ref_and_q_vs_states = ref_and_q_vs_states.T.drop_duplicates().T
    ref_and_q_vs_states['Ref. (std)'] = compute_ffbsi_stds(means_ref, state_seqs)
    ref_and_q_vs_states[f'MAE marginals ({pretty_names[0]})'] = compute_mae_marginals(means_ref[-1], smooth_results, 0)
    ref_and_q_vs_states[f'MAE marginals ({pretty_names[1]})'] = compute_mae_marginals(means_ref[-1], smooth_results, 1)

    end_table = ref_and_q_vs_states[['Ref.',
                                    'Ref. (std)',
                                    f'Additive |Ref.-{pretty_names[0]}|',
                                    f'Additive |Ref.-{pretty_names[1]}|',
                                    f'MAE marginals ({pretty_names[0]})',
                                    f'MAE marginals ({pretty_names[1]})']] / train_args.state_dim

    # print(end_table.to_latex(float_format="%.2f" ))
    # print(end_table.to_markdown())
    print(end_table)
    q_vs_ref_marginals = pd.concat(q_vs_ref_marginals_all, keys=pretty_names)
    q_vs_ref_marginals.columns = ['Sequence', 'n', 'Value']
    q_vs_ref_marginals = q_vs_ref_marginals.reset_index(level=0).reset_index(drop=True)
    q_vs_ref_marginals.columns = ['Method', 'Sequence', 'n', 'Value']

    q_vs_ref_additive = pd.concat(q_vs_ref_additive_all, keys=pretty_names)
    q_vs_ref_additive.columns = ['Sequence', 'n', 'Value']
    q_vs_ref_additive = q_vs_ref_additive.reset_index(level=0).reset_index(drop=True) 
    q_vs_ref_additive.columns = ['Method', 'Sequence', 'n', 'Value']


    import numpy as np
    sns.lineplot(data=q_vs_ref_marginals, x='n', y='Value', hue='Method', style='Sequence')
    # plt.title('Marginal 1-norm error against FFBSi')
    plt.savefig(os.path.join(eval_dir, f'marginal_errors'))
    plt.close()

    sns.lineplot(data=q_vs_ref_additive, x='n', y='Value',hue='Method')
    sns.lineplot(data=q_vs_ref_additive, x='n', y='Value',hue='Method',style='Sequence',alpha=0.3, legend=False)
    plt.ylabel('Additive error')

    # sns.lineplot(data=q_vs_ref_additive, x='Timestep', y='Value',hue='Method', ax=ax)
    # plt.title('Additive 1-norm error against FFBSi')

    plt.savefig(os.path.join(eval_dir, f'additive_errors'))
    plt.close()

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


