#%%
from typing import NamedTuple
import haiku as hk 
import jax 
import jax.numpy as jnp
import backward_ica.stats.hmm as hmm
from backward_ica.stats import set_parametrization
import backward_ica.variational as variational
import backward_ica.utils as utils 
import backward_ica.stats.smc as smc
import seaborn as sns
import os 
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import pandas as pd
from pandas.plotting import table
import dill 
from backward_ica.elbos import BackwardLinearELBO
import pickle

utils.enable_x64(True)

exp_dir = 'experiments/p_chaotic_rnn/2022_10_20__11_53_44'

method_name = 'johnson_backward'
                
pretty_name = 'Johnson Backward'

seq_length = 2000
num_particles = 1000
num_smooth_particles = 1000
num_seqs = 1
load = False
metrics = True
plot_sequences = True
recompute_marginals = False
profile = False
filter_rmse = True
visualize_init = False
lag = None
ref_type = 'states'
num_slices = 10

train_args = utils.load_args('train_args', os.path.join(exp_dir, method_name))
key_theta = jax.random.PRNGKey(train_args.seed_theta)
train_args.num_particles = num_particles
train_args.num_smooth_particles = num_smooth_particles


train_args.loaded_data = (os.path.join(utils.chaotic_rnn_base_dir, 'x_data.npy'), 
                        os.path.join(utils.chaotic_rnn_base_dir,'y_data.npy'))
train_args.num_seqs = 1
train_args.seq_length = seq_length
    
set_parametrization(train_args)

eval_dir = os.path.join(exp_dir, f'eval_{method_name}')
os.makedirs(eval_dir, exist_ok=True)
from time import time

# shutil.rmtree(eval_dir)


    
p = hmm.get_generative_model(train_args)
theta_star = utils.load_params('theta', os.path.join(exp_dir, method_name))



key_theta, key_gen, key_ffbsi = jax.random.split(key_theta,3)
if train_args.loaded_data: 
    state_seqs, obs_seqs = p.sample_multiple_sequences(key_gen, theta_star, train_args.num_seqs, seq_length, train_args.single_split_seq, train_args.loaded_data)
else:
    state_seqs, obs_seqs = p.sample_multiple_sequences(key_gen, theta_star, num_seqs, seq_length)

    # print(state_seqs.shape)
    # print(state_seqs.dtype)

keys_ffbsi = jax.random.split(key_theta, num_seqs)
filt_results, smooth_results = [], []


class ExternalVariationalFamily():

    def __init__(self, save_dir, length=None):
        self.means_filt_q = jnp.load(os.path.join(save_dir, 'filter_means.npy'))[jnp.newaxis,:length]
        self.covs_filt_q = jnp.load(os.path.join(save_dir, 'filter_covs.npy'))[jnp.newaxis,:length]
        with open(os.path.join(save_dir, 'smoothed_stats.pickle'), 'rb') as f: 
            smoothed_means, smoothed_covs = pickle.load(f)
        self.means_smooth_q_list = [smoothed_means[i] for i in range(length)]
        self.covs_smooth_q_list = [smoothed_covs[i] for i in range(length)]

    def get_filt_means_and_covs(self):
        return (self.means_filt_q, self.covs_filt_q)
    
    def get_smooth_means_and_covs(self):
        return (self.means_smooth_q_list[-1][jnp.newaxis,:], self.covs_smooth_q_list[-1][jnp.newaxis,:])

    def smooth_seq_at_multiple_timesteps(self, obs_seq, phi, slices):

        smoothed_means = [self.means_smooth_q_list[timestep-1] for timestep in slices]
        smoothed_covs = [self.covs_smooth_q_list[timestep-1] for timestep in slices]

        return (smoothed_means, smoothed_covs)



if 'external' in method_name:
    q = ExternalVariationalFamily(utils.chaotic_rnn_base_dir, seq_length)

    filt_results.append(q.get_filt_means_and_covs())
    smooth_results.append(q.get_smooth_means_and_covs())

    phi = None
else: 
    method_dir = os.path.join(exp_dir, method_name)
    args = utils.load_args('train_args', method_dir)

    key_phi = jax.random.PRNGKey(args.seed_phi)

    key_phi, key_filt_q, key_smooth_q = jax.random.split(key_phi, 3)
    keys_smooth_q = jax.random.split(key_smooth_q, num_seqs)

    q = variational.get_variational_model(args, p)

    if visualize_init: 
        phi = q.get_random_params(key_phi, args)
    else:
        phi = utils.load_params('phi', method_dir)


    means_filt_q, covs_filt_q = jax.vmap(q.filt_seq, in_axes=(0, None))(obs_seqs, phi)
    means_smooth_q, covs_smooth_q = jax.vmap(q.smooth_seq, in_axes=(0,None,None))(obs_seqs, phi, lag)
    filt_results.append((means_filt_q, covs_filt_q))
    smooth_results.append((means_smooth_q, covs_smooth_q))





if filter_rmse: 
    filt_rmse_q = jnp.mean(jnp.sqrt(jnp.mean((means_filt_q - state_seqs)**2, axis=-1)))
    print(f'Filter RMSE {pretty_name}:', filt_rmse_q)
    smooth_rmse_q = jnp.mean(jnp.sqrt(jnp.mean((means_smooth_q - state_seqs)**2, axis=-1)))
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
        means_q, covs_q = results[0]
        for seq_nb in range(num_seqs):
            fig, axes = plt.subplots(train_args.state_dim, 1, sharey='row', figsize=(30,30))
            plt.autoscale(True)
            plt.tight_layout()
            # if len(method_names) > 1: axes = np.atleast_2d(axes)
            name = f'{task_name}_seq_{seq_nb}'
            for dim_nb in range(train_args.state_dim):
                axes[dim_nb].plot(range(len(state_seqs[seq_nb])), state_seqs[seq_nb,:,dim_nb], color='green', linestyle='dashed', label='True state')
                utils.plot_relative_errors_1D(axes[dim_nb], means_q[seq_nb,:,dim_nb], covs_q[seq_nb,:,dim_nb,dim_nb], color='red', alpha=0.2, label=f'{method_name}')
                axes[dim_nb].legend()
            plt.savefig(os.path.join(eval_dir, name))
            plt.close()


def recompute_marginals_func(results):
    means_smc = state_seqs
    means_q = means_smooth_q
    def compute_marginal(means_smc, means_q):
        return jnp.linalg.norm(means_q - means_smc, ord=1, axis=1)
    return jax.vmap(compute_marginal)(means_smc, means_q)

def compute_ffbsi_stds(means_smc, state_seqs):
    def compute_ref_vs_states(means_smc, state_seq):
        ref_vs_states = jnp.linalg.norm(means_smc[-1] - state_seq, ord=1, axis=1)
        return jnp.var(ref_vs_states, axis=0)

    return jax.vmap(compute_ref_vs_states)(means_smc, state_seqs)

def compute_mae_marginals(means_ref):
    def compute_marginal_mae(means_smc, means_q):
        return jnp.mean(jnp.linalg.norm(means_q - means_smc, ord=1, axis=1), axis=0)
    return jax.vmap(compute_marginal_mae)(means_ref, means_smooth_q)


def eval_smoothing_single_seq(state_seq, obs_seq, means_ref, slices):


    means_q = q.smooth_seq_at_multiple_timesteps(obs_seq, phi, slices)[0]

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
       

eval_smoothing = jax.vmap(eval_smoothing_single_seq, in_axes=(0,0,0, None))

if metrics: 

    slice_length = len(obs_seqs[0]) // num_slices
    slices = jnp.array(list(range(0, len(obs_seqs[0])+1, slice_length)))[1:]


    means_ref = [state_seqs[:,:timestep] for timestep in slices]

    print(f'Evaluating {method_name}')

    ref_and_q_vs_states, q_vs_ref_marginals, q_vs_ref_additive = eval_smoothing(state_seqs, obs_seqs, means_ref, slices)

    with open(os.path.join(eval_dir, f'eval_{method_name}.dill'), 'wb') as f:
        dill.dump((ref_and_q_vs_states, q_vs_ref_marginals, q_vs_ref_additive), f)
    print('Done.')


    ref_and_q_vs_states = pd.DataFrame(ref_and_q_vs_states, columns = ['Ref.', f'{pretty_name}', f'Additive |Ref.-{pretty_name}|'])

    q_vs_ref_marginals = pd.DataFrame(data=q_vs_ref_marginals.T / train_args.state_dim).unstack().reset_index(name='value')
    q_vs_ref_additive = pd.DataFrame(index=slices, data=q_vs_ref_additive.T / train_args.state_dim).unstack().reset_index(name='value')


    ref_and_q_vs_states = ref_and_q_vs_states.T.drop_duplicates().T
    ref_and_q_vs_states['Ref. (std)'] = compute_ffbsi_stds(means_ref, state_seqs)
    ref_and_q_vs_states[f'MAE marginals ({pretty_name})'] = compute_mae_marginals(means_ref[-1])

    end_table = ref_and_q_vs_states[['Ref.',
                                    'Ref. (std)',
                                    f'Additive |Ref.-{pretty_name}|',
                                    f'MAE marginals ({pretty_name})']] / train_args.state_dim

    # print(end_table.to_latex(float_format="%.2f" ))
    # print(end_table.to_markdown())
    print(end_table)
    q_vs_ref_marginals.columns = ['Sequence', 'n', 'Value']
    q_vs_ref_marginals = q_vs_ref_marginals.reset_index(level=0).reset_index(drop=True)
    q_vs_ref_marginals.columns = ['Method', 'Sequence', 'n', 'Value']

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



    plt.savefig(os.path.join(eval_dir, f'additive_errors'))
    plt.close()
