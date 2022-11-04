#%%
import jax 
import jax.numpy as jnp
import backward_ica.stats.hmm as hmm
from backward_ica.stats import set_parametrization
import backward_ica.variational as variational
import backward_ica.utils as utils 
import os 
import matplotlib.pyplot as plt
from pandas.plotting import table
import dill 
import pickle

utils.enable_x64(True)

def main(args, method_name):
        
    eval_dir = os.path.join(args.exp_dir, 'evals', method_name)
    os.makedirs(eval_dir, exist_ok=True)
    utils.save_args(args, 'args', eval_dir)

    if method_name == 'johnson_backward':
        pretty_name = 'Conjugate Backward'

    elif method_name == 'johnson_forward':
        pretty_name = 'Conjugate Forward'

    elif method_name == 'neural_backward_linear':
        pretty_name = 'GRU Backward'

    elif method_name == 'external_campbell':
        pretty_name = 'Campbell'
    
    elif method_name == 'ffbsi':
        pretty_name = 'FFBSi'

    elif method_name.split('_')[0] == 'linear':
        pretty_name = 'Linear'
        epoch_nb = int(method_name.split('_')[1])
        method_name = 'linear'


    

    metrics = True
    visualize_init = False
    lag = None


    data_args = utils.load_args('args', args.exp_dir)
    seq_length = data_args.seq_length
    data_args.num_particles = args.n_bootstrap 
    data_args.num_smooth_particles = args.n_ffbsi 

    p = hmm.get_generative_model(data_args)
    theta_star = utils.load_params('theta_star', args.exp_dir)

    num_seqs = args.num_seqs
    if num_seqs != 1: 
        num_seqs = 1 if args.load_seq else num_seqs
        state_seqs, obs_seqs = p.sample_multiple_sequences(jax.random.PRNGKey(0), 
                                                        theta_star,
                                                        num_seqs=num_seqs, 
                                                        seq_length=seq_length,
                                                        single_split_seq=False,
                                                        load_from=data_args.load_from,
                                                        loaded_seq=data_args.loaded_seq)
                                                

    else: 
        obs_seqs = jnp.load(os.path.join(args.exp_dir, 'obs_seqs.npy'))
        state_seqs = jnp.load(os.path.join(args.exp_dir, 'state_seqs.npy'))

    set_parametrization(data_args)



        

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

    elif 'ffbsi' in method_name or method_name == 'linear_0':
        pass        
    else: 
        method_dir = os.path.join(args.exp_dir, method_name)

        train_args = utils.load_args('args', method_dir)
        train_args.state_dim, train_args.obs_dim = p.state_dim, p.obs_dim

        key_phi = jax.random.PRNGKey(train_args.seed)

        key_phi, key_filt_q, key_smooth_q = jax.random.split(key_phi, 3)
        keys_smooth_q = jax.random.split(key_smooth_q, num_seqs)

        q = variational.get_variational_model(train_args, p)

        if visualize_init: 
            phi = q.get_random_params(key_phi, train_args)
        else:
            phi = utils.load_params('phi', method_dir)
            if epoch_nb != 'best':
                phi = phi[epoch_nb]
        if args.plot: 
            means_filt_q, covs_filt_q = jax.vmap(q.filt_seq, in_axes=(0, None))(obs_seqs, phi)
            means_smooth_q, covs_smooth_q = jax.vmap(q.smooth_seq, in_axes=(0,None,None))(obs_seqs, phi, lag)

    if args.plot:
        filt_results.append((means_filt_q, covs_filt_q))
        smooth_results.append((means_smooth_q, covs_smooth_q))


    if args.filter_rmse: 
        filt_rmse_q = jnp.mean(jnp.sqrt(jnp.mean((means_filt_q - state_seqs)**2, axis=-1)))
        print(f'Filter RMSE {pretty_name}:', filt_rmse_q)
        smooth_rmse_q = jnp.mean(jnp.sqrt(jnp.mean((means_smooth_q - state_seqs)**2, axis=-1)))
        print(f'Smoothing RMSE {pretty_name}:', smooth_rmse_q)
        print('-----')
        if method_name == 'external_campbell':
            filt_rmses_campbell = jnp.load(os.path.join(data_args.load_from, 'filter_RMSEs.npy'))[:,-1]
            print('Filter RMSE campbell external:', jnp.mean(filt_rmses_campbell))
    #%%
    if args.plot: 
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

        

    if metrics: 

        slice_length = len(obs_seqs[0]) // args.n_slices
        slices = jnp.array(list(range(0, len(obs_seqs[0])+1, slice_length)))[1:]
        means_ref = [state_seqs[:,:timestep] for timestep in slices]

        print(f'Evaluating {method_name}')
        if 'ffbsi' in method_name: 
            key = jax.random.PRNGKey(0)
            means_q = jax.vmap(p.smooth_seq_at_multiple_timesteps, in_axes=(0,0,None,None))(jax.random.split(key, len(obs_seqs)), 
                                                                                            obs_seqs, 
                                                                                            theta_star, 
                                                                                            slices)[0]
        elif method_name == 'linear_0':
            means_q = jax.vmap(p.smooth_seq_at_multiple_timesteps, in_axes=(0,None,None))(obs_seqs, theta_star, slices)[0]
        else: 
            means_q = jax.vmap(q.smooth_seq_at_multiple_timesteps, in_axes=(0,None,None))(obs_seqs, phi, slices)[0]

        with open(os.path.join(eval_dir, 'eval.dill'), 'wb') as f:
            dill.dump((means_q, means_ref, slices), f)
        print('Done.')

if __name__ == '__main__':

    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir',type=str, default='')
    parser.add_argument('--models', type=str, nargs='+', default=['linear_10','linear_15','linear_195'])
    parser.add_argument('--n_slices', type=int, default=250)     
    parser.add_argument('--rmse', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--num_seqs', type=int, default=1)
    parser.add_argument('--n_bootstrap', type=int, default=10000)
    parser.add_argument('--n_ffbsi', type=int, default=2000)

    args = parser.parse_args()
    args.load_seq = False
    args.plot = False 
    args.filter_rmse = False 
    for method_name in args.models:
        main(args,
            method_name)
    