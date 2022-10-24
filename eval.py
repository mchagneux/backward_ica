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

def main(exp_dir, method_name, num_slices):
        

    if method_name == 'johnson_backward':
        pretty_name = 'Conjugate Backward'
    elif method_name == 'johnson_forward':
        pretty_name = 'Conjugate Forward'
    elif method_name == 'neural_backward_linear':
        pretty_name = 'GRU Backward'
    elif method_name == 'external_campbell':
        pretty_name = 'Campbell'

    metrics = True
    plot_sequences = True
    filter_rmse = True
    visualize_init = False
    lag = None
    seq_length = 500
    num_seqs = 1

    data_args = utils.load_args('args', exp_dir)
    p = hmm.get_generative_model(utils.load_args('args', exp_dir))
    theta_star = utils.load_params('theta_star', exp_dir)

    if data_args.model == 'chaotic_rnn':
        obs_seqs = jnp.load(os.path.join(exp_dir, 'obs_seqs.npy'))
        state_seqs = jnp.load(os.path.join(exp_dir, 'state_seqs.npy'))
    else: 
        state_seqs, obs_seqs = p.sample_multiple_sequences(jax.random.PRNGKey(data_args.seed), 
                                                        theta_star,
                                                        num_seqs=num_seqs, 
                                                        seq_length=seq_length,
                                                        single_split_seq=False,
                                                        load_from='')

    set_parametrization(data_args)

    eval_dir = os.path.join(exp_dir, method_name, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    from time import time

        

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
        q = ExternalVariationalFamily(data_args.load_from, seq_length)

        means_filt_q, covs_filt_q = q.get_filt_means_and_covs()
        means_smooth_q, covs_smooth_q = q.get_smooth_means_and_covs()



        phi = None
    else: 
        method_dir = os.path.join(exp_dir, method_name)
        args = utils.load_args('args', method_dir)
        args.state_dim, args.obs_dim = p.state_dim, p.obs_dim

        key_phi = jax.random.PRNGKey(args.seed)

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
        if method_name == 'external_campbell':
            filt_rmses_campbell = jnp.load(os.path.join(data_args.load_from, 'filter_RMSEs.npy'))[:,-1]
            print(f'Filter RMSE campbell external:', jnp.mean(filt_rmses_campbell))

    #%%

    if plot_sequences: 
        colors = ['blue',
                'red']
        print('Plotting individual sequences...')
        for task_name, results in zip(['filtering','smoothing'], [filt_results, smooth_results]): 
            means_q, covs_q = results[0]
            for seq_nb in range(num_seqs):
                fig, axes = plt.subplots(data_args.state_dim, 1, sharey='row', figsize=(30,30))
                plt.autoscale(True)
                plt.tight_layout()
                # if len(method_names) > 1: axes = np.atleast_2d(axes)
                name = f'{task_name}_seq_{seq_nb}'
                for dim_nb in range(p.state_dim):
                    axes[dim_nb].plot(range(len(state_seqs[seq_nb])), state_seqs[seq_nb,:,dim_nb], color='green', linestyle='dashed', label='True state')
                    utils.plot_relative_errors_1D(axes[dim_nb], means_q[seq_nb,:,dim_nb], covs_q[seq_nb,:,dim_nb,dim_nb], color='red', alpha=0.2, label=f'{pretty_name}')
                    axes[dim_nb].legend()
                plt.savefig(os.path.join(eval_dir, name))
                plt.close()


    def eval_smoothing_single_seq(means_ref, obs_seq, slices):


        means_q = q.smooth_seq_at_multiple_timesteps(obs_seq, phi, slices)[0]

        q_vs_ref_marginals = jnp.linalg.norm((means_q[-1] - means_ref[-1]), ord=1, axis=1)[slices]
        
        q_vs_ref_additive = []
        for means_ref_n, means_q_n in zip(means_ref, means_q):
            q_vs_ref_additive.append(jnp.linalg.norm(jnp.sum(means_ref_n - means_q_n, axis=0), ord=1))
        q_vs_ref_additive = jnp.array(q_vs_ref_additive)

        return q_vs_ref_marginals, q_vs_ref_additive
        

    eval_smoothing = jax.vmap(eval_smoothing_single_seq, in_axes=(0,0, None))

    if metrics: 

        slice_length = len(obs_seqs[0]) // num_slices
        slices = jnp.array(list(range(0, len(obs_seqs[0])+1, slice_length)))[1:]


        means_ref = [state_seqs[:,:timestep] for timestep in slices]

        print(f'Evaluating {method_name}')

        q_vs_ref_marginals, q_vs_ref_additive = eval_smoothing(state_seqs, obs_seqs, slices)

        with open(os.path.join(eval_dir, 'eval.dill'), 'wb') as f:
            dill.dump((q_vs_ref_marginals, q_vs_ref_additive), f)
        print('Done.')

if __name__ == '__main__':

    exp_dir = 'experiments/p_chaotic_rnn/2022_10_24__17_25_51'
    method_names = ['external_campbell', 'johnson_backward', 'johnson_forward', 'neural_backward_linear']
    for method_name in method_names:
        main(exp_dir, method_name, 50)
    # import argparse 
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--exp_dir',type=str,default='')
    # parser.add_argument('--num_slices',type=int,default=10)
    