#%%
import argparse
import haiku as hk 
import jax 
import jax.numpy as jnp
import backward_ica.hmm as hmm 
import backward_ica.utils as utils 
import backward_ica.smc as smc
from backward_ica.svi import GeneralBackwardELBO, LinearGaussianELBO, OnlineGeneralBackwardELBO
import seaborn as sns
import os 
import pandas as pd
import matplotlib.pyplot as plt
import random
import pandas as pd
import math
gaussian_distance = hmm.Gaussian.KL



def plot():
    
    get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
    colors = get_colors(num_indices)
    for seq_nb in range(num_seqs):
        
        samples_seq = samples_seqs[seq_nb]
        backwd_state_seq = utils.tree_get_idx(seq_nb, backwd_state_seqs)

        for time_idx in range(0, seq_length, seq_length // 5):
            fig, axes = plt.subplots(state_dim, 1, figsize=(20,30))

            samples_old = samples_seq[time_idx]
            samples_new = samples_seq[time_idx+1]
            key, subkey = jax.random.split(key, 2)

            random_indices = jax.random.choice(subkey, 
                                            jnp.arange(0, len(samples_new)), 
                                            shape=(num_indices,),
                                            replace=False)
            for dim_nb in range(state_dim):

                sns.histplot(samples_old[:,dim_nb], 
                            ax=axes[dim_nb], 
                            stat='density',
                            label=f'$\\xi_t^j[{dim_nb}]$',
                            color='grey')

            for num_idx in range(num_indices):
                random_idx = random_indices[num_idx]

                new_sample_i = samples_new[random_idx]
                # weights = weights_seq[time_idx][random_idx]

                backwd_params = q.backwd_kernel.map(new_sample_i, utils.tree_get_idx(time_idx, backwd_state_seq))

                for dim_nb in range(state_dim):
                    samples_x = samples_old[:,dim_nb]
                    range_x = jnp.linspace(samples_x.min(), samples_x.max(), 1000)
                    mu, sigma = backwd_params.mean[dim_nb], backwd_params.scale.cov[dim_nb, dim_nb]
                    backwd_pdf = lambda x: hmm.gaussian_pdf(x, mu, sigma)
                                                            
                    axes[dim_nb].plot(range_x, 
                                    backwd_pdf(range_x), 
                                    label=f'$q(x_t[{dim_nb}] | \\xi_{{t+1}}^{{{random_idx}}})$', 
                                    color=colors[num_idx])
                    axes[dim_nb].legend()
                # sns.histplot(weights, ax=axes[state_dim], label=f'$\\omega_t^{{{random_idx}}}j$', color=colors[num_idx])
                # axes[state_dim].legend()


            plt.suptitle(f'Sequence {seq_nb}, time {time_idx}, (difference online/offline ELBO {jnp.abs(online_mc_elbo_values[seq_nb] - offline_mc_elbo_values[seq_nb]):.2f})')
            plt.autoscale(True)
            plt.tight_layout()

            # sns.pairplot(data=samples, 
            #                 diag_kws={'weights':weights}, 
            #                 plot_kws={'weights':weights}, kind="kde")
            plt.savefig(os.path.join(save_dir, f'seq_{seq_nb}_time_{time_idx}'))
            plt.close()

def main(args):

    key = jax.random.PRNGKey(0)
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    num_samples = args.num_samples
    num_seqs = args.num_seqs
    seq_length = args.seq_length
    trained_model = args.trained_model

    if trained_model: 

        args_p = utils.load_args('train_args',os.path.join(args.exp_dir, args.method_name))

        method_dir = os.path.join(args.exp_dir, args.method_name)
        args_q = utils.load_args('train_args', method_dir)
        args_p.linear = args_q.split('_')[0] == 'linear'
        state_dim, obs_dim = args_p.state_dim, args_p.obs_dim


    else: 
        state_dim, obs_dim = args.state_dim, args.obs_dim

        args_p = argparse.Namespace()
        args_p.state_dim, args_p.obs_dim = args.state_dim, args.obs_dim

        args_p.linear = True
        args_p.layers = ()
        args_p.slope = 0
        args_p.num_particles = 1000 
        args_p.num_smooth_particles = 1000
        args_p.transition_bias = False 
        args_p.range_transition_map_params = args.range_transition_map_params
        args_p.transition_matrix_conditionning = 'diagonal'
        args_p.injective = True

        args_p.parametrization = 'cov_chol'
        args_p.default_prior_mean = 0.0
        args_p.default_prior_base_scale = math.sqrt(1e-2)
        args_p.default_transition_base_scale = math.sqrt(1e-2)
        args_p.default_emission_base_scale = math.sqrt(1e-2)
        args_p.default_transition_bias = 0

        args_q = argparse.Namespace()
        args_q.q_version = 'linear'
        args_q.transition_matrix_conditionning = 'diagonal'
        args_q.range_transition_map_params = args.range_transition_map_params
        args_q.transition_bias = False
        args_q.emission_bias = False 
        args_q.update_layers = (8,)
        args_q.backwd_layers = False
        args_q.explicit_proposal = False 

    utils.set_parametrization(args_p)


    if args_p.linear:
        p = hmm.LinearGaussianHMM(state_dim, 
                                obs_dim, 
                                args_p.transition_matrix_conditionning, 
                                args_p.range_transition_map_params, 
                                args_p.transition_bias)

    else: 

        p = hmm.NonLinearHMM.linear_transition_with_nonlinear_emission(args_p) # specify the structure of the true model

    if 'linear' in args_q.q_version:

        q = hmm.LinearGaussianHMM(state_dim=state_dim, 
                                obs_dim=obs_dim, 
                                transition_matrix_conditionning=args_q.transition_matrix_conditionning,
                                range_transition_map_params=args_q.range_transition_map_params,
                                transition_bias=args_q.transition_bias,
                                emission_bias=args_q.emission_bias) 

    elif 'johnson' in args_q.q_version:
        q = hmm.JohnsonBackwardSmoother(transition_kernel=p.transition_kernel,
                                        obs_dim=obs_dim, 
                                        update_layers=args_q.update_layers,
                                        explicit_proposal='explicit_proposal' in args_q.q_version)


    else:
        q = hmm.GeneralBackwardSmoother(state_dim=state_dim, 
                                        obs_dim=obs_dim, 
                                        update_layers=args_q.update_layers,
                                        backwd_layers=args_q.backwd_map_layers)


    if args.trained_model:
        theta_star = utils.load_params('theta', os.path.join(args.exp_dir, args.method_name))
        phi = utils.load_params('phi', args.method_dir)[1]

    else: 
        key, subkey_theta, subkey_phi = jax.random.split(key, 3)
        theta_star = p.get_random_params(subkey_theta, args_p)
        phi = q.get_random_params(subkey_phi, args_p)



    key, subkey = jax.random.split(key, 2)

    state_seqs, obs_seqs = p.sample_multiple_sequences(subkey, theta_star, num_seqs, seq_length)

    # normalizer = lambda x: jnp.mean(jnp.exp(x))
    normalizer = smc.exp_and_normalize

    if args_p.linear:
        closed_form_elbo = jax.jit(jax.vmap(lambda obs_seq: LinearGaussianELBO(p,q)(obs_seq, p.format_params(theta_star), q.format_params(phi))))

    offline_mc_elbo = jax.jit(jax.vmap(lambda key, obs_seq: GeneralBackwardELBO(p, q, num_samples)(key, obs_seq, p.format_params(theta_star), q.format_params(phi))))
    online_mc_elbo = jax.jit(jax.vmap(lambda key, obs_seq: OnlineGeneralBackwardELBO(p, q, normalizer, num_samples)(key, obs_seq, p.format_params(theta_star), q.format_params(phi))))

    keys = jax.random.split(key, num_seqs)

        
    offline_mc_elbo_values = offline_mc_elbo(keys, obs_seqs)
    online_mc_elbo_values, (samples_seqs, weights_seqs, filt_state_seqs, backwd_state_seqs) = online_mc_elbo(keys, obs_seqs)


    offline_mc_elbo_values /= seq_length 
    online_mc_elbo_values /= seq_length

    if args_p.linear: 
        true_elbo_values = closed_form_elbo(obs_seqs) / seq_length
        errors_online = jnp.abs(true_elbo_values - online_mc_elbo_values)
        errors_offline = jnp.abs(true_elbo_values - offline_mc_elbo_values)
        errors = pd.DataFrame({'Online':errors_online, 
                                'Offline':errors_offline}).unstack().reset_index()
        errors.columns = ['Method', 'Sequence nb', 'Error']
        sns.boxplot(data=errors, x='Method', y='Error')
        plt.savefig(os.path.join(save_dir, 'errors_boxplot.pdf'),format='pdf')
        plt.close()
        # sns.boxplot(errors)
    #%%

    def distance_backwd_to_filt(sample, filt_state, backwd_state):
        return gaussian_distance(q.backwd_kernel.map(sample, backwd_state), filt_state.out)
        
    def distances_backwd_to_filt(samples_seqs, filt_state_seqs, backwd_state_seqs):

        def distances_for_seq(samples_seq, filt_state_seq, backwd_state_seq):
            samples_seq = utils.tree_dropfirst(samples_seq)
            filt_state_seq = utils.tree_droplast(filt_state_seq)

            return jax.vmap(jax.vmap(distance_backwd_to_filt, in_axes=(0,None,None)), in_axes=(0,0,0))(samples_seq, filt_state_seq, backwd_state_seq)

        distances = jax.vmap(distances_for_seq)(samples_seqs, filt_state_seqs, backwd_state_seqs)
        distances_list = []
        for seq_nb in range(distances.shape[0]):
            distances_seq = pd.DataFrame(distances[seq_nb]).unstack().reset_index()
            distances_seq.columns = ['Sample_nb','Timestep','Value']
            distances_list.append(distances_seq)
        return distances_list

    distances = distances_backwd_to_filt(samples_seqs, filt_state_seqs, backwd_state_seqs)

    fig, axes = plt.subplots(num_seqs, 1, figsize=(20,20))
    plt.tight_layout()
    plt.autoscale()
    for seq_nb in range(num_seqs):
        sns.boxplot(data=distances[seq_nb], x='Timestep',y='Value', ax=axes[seq_nb])
    plt.savefig(os.path.join(save_dir, 'kullback_distances.pdf'),format='pdf')
    plt.close()
# sns.boxplot(data=distances, x='Timestep', y='Value', hue='Sequence_nb')
#
#%%
        # plt.savefig('')


# get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
# g = sns.FacetGrid(errors, row="smoker", col="time", margin_titles=True)

# sns.kdeplot(offline_errors, olor='red')
# sns.kdeplot(online_errors, color='blue')

#%%

if __name__ == '__main__':

    args = argparse.Namespace()
    
    args.state_dim = 10
    args.obs_dim = 10
    args.num_samples = 1000
    args.num_seqs = 10
    args.seq_length = 50
    args.trained_model = False
    args.range_transition_map_params = (0.99,1)
    args.save_dir = 'experiments/tests/online/tests'
    args.exp_dir = 'experiments/p_nonlinear/2022_07_27__12_21_27'
    args.method_name = 'johnson_freeze__theta'

    main(args)

# weights = jnp.exp(log_weights) / num_samples

# for t, weights_t in enumerate(weights):s
#     g = sns.displot(weights_t.flatten(), bins=100, kind='hist')
#     g.savefig(os.path.join(save_dir, f'{t}'))

