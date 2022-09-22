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


def get_sampler_and_elbos(key, args, args_p, args_q):


    key, *subkeys = jax.random.split(key, 3)
    p, theta_star = utils.get_generative_model(args_p, subkeys[0])
    q, phi = utils.get_variational_model(args_q, p, subkeys[1])


    if args.trained_model:
        theta_star = utils.load_params('theta', os.path.join(args.exp_dir, args.method_name))
        phi = utils.load_params('phi', args.method_dir)[1]


    # normalizer = lambda x: jnp.mean(jnp.exp(x))
    normalizer = smc.exp_and_normalize


    if args_p.p_version == 'linear':
        closed_form_elbo = jax.jit(jax.vmap(lambda obs_seq: LinearGaussianELBO(p, q)(obs_seq, 
                                                                            p.format_params(theta_star), 
                                                                            q.format_params(phi))))
    else: 
        closed_form_elbo = lambda obs_seq: 0

    keys = jax.random.split(key, args.num_seqs)

    offline_mc_elbo = jax.vmap(in_axes=(0,0,None), fun=lambda key, obs_seq, num_samples: GeneralBackwardELBO(p, q, num_samples)(
                                                                                                        key, 
                                                                                                        obs_seq, 
                                                                                                        p.format_params(theta_star), 
                                                                                                        q.format_params(phi)))
    
    online_mc_elbo = jax.vmap(in_axes=(0,0,None), fun=lambda key, obs_seq, num_samples: OnlineGeneralBackwardELBO(p, q, normalizer, num_samples)(
                                                                                                                    key, 
                                                                                                                    obs_seq, 
                                                                                                                    p.format_params(theta_star), 
                                                                                                                    q.format_params(phi)))
    def mc_elbos(obs_seqs, num_samples):
        offline_value = offline_mc_elbo(keys, obs_seqs, num_samples) 
        online_value, (samples_seqs, weights_seqs, filt_state_seqs, backwd_state_seqs) = online_mc_elbo(keys, obs_seqs, num_samples) 
        return offline_value / len(obs_seqs[0]), online_value / len(obs_seqs[0]), (samples_seqs, weights_seqs, filt_state_seqs, backwd_state_seqs)

    def true_elbo(obs_seqs):
        return closed_form_elbo(obs_seqs) / len(obs_seqs[0])

    def sampler(key):
        return p.sample_multiple_sequences(key, theta_star, args.num_seqs, args.seq_length)

    return sampler, mc_elbos, true_elbo
    
def main(args, args_p, args_q):

    key = jax.random.PRNGKey(0)


    utils.set_parametrization(args_p)

    if args.evolution_wrt_seq_length != -1: 


        key, *subkeys = jax.random.split(key, 3)

        sampler, mc_elbos, true_elbo = get_sampler_and_elbos(subkeys[0], args, args_p, args_q)

        mc_elbos_jit = jax.jit(lambda obs_seqs: mc_elbos(obs_seqs, args.num_samples))

        state_seqs, obs_seqs = sampler(subkeys[1])

        key, *subkeys = jax.random.split(key, args.num_seqs)
        
        true_values_list = dict()
        offline_values_list = dict()
        online_values_list = dict()

        for stopping_point in range(args.evolution_wrt_seq_length, args.seq_length, args.evolution_wrt_seq_length):

            true_values = true_elbo(obs_seqs[:,:stopping_point])
            offline_values, online_values = mc_elbos_jit(obs_seqs[:,:stopping_point])[:-1]

            true_values_list[stopping_point] = true_values
            offline_values_list[stopping_point] = offline_values
            online_values_list[stopping_point] = online_values


        true_values = true_elbo(obs_seqs)
        offline_values, online_values, (samples_seqs, weights_seqs, filt_state_seqs, backwd_state_seqs) = mc_elbos_jit(obs_seqs)

        true_values_list[args.seq_length] = true_values
        offline_values_list[args.seq_length] = offline_values
        online_values_list[args.seq_length] = online_values

        errors_online = dict()
        errors_offline = dict()

        for k in true_values_list.keys():
            errors_online_k = jnp.abs(true_values_list[k] - online_values_list[k])
            errors_online[k] = {k:v.tolist() for k,v in enumerate(errors_online_k)}
            errors_offline_k  = jnp.abs(true_values_list[k] - offline_values_list[k])
            errors_offline[k] = {k:v.tolist() for k,v in enumerate(errors_offline_k)}

        errors_online = pd.DataFrame(errors_online).apply(pd.Series).unstack().reset_index()
        errors_online['Method'] = 'Online'
        errors_offline = pd.DataFrame(errors_offline).apply(pd.Series).unstack().reset_index()   
        errors_offline['Method'] = 'Offline'
        errors = pd.concat([errors_online, errors_offline], axis=0).reset_index().drop(columns='index')

        errors.columns = ['Seq length', 'Seq nb', 'Value', 'Method']
        errors.to_csv(os.path.join(args.save_dir, 'errors_wrt_length.csv'))

        sns.violinplot(data=errors, 
                    x='Seq length', 
                    y='Value',
                    hue='Method',
                    inner='point')

        plt.savefig(os.path.join(args.save_dir, 'errors_wrt_length_violinplot.pdf'), format='pdf')
        plt.close()

    if args.evolution_wrt_num_samples != -1:

        key, *subkeys = jax.random.split(key, 3)

        sampler, mc_elbos, true_elbo = get_sampler_and_elbos(subkeys[0], args, args_p, args_q)

        state_seqs, obs_seqs = sampler(subkeys[1])

        offline_values_list = dict()
        online_values_list = dict()

        true_values = true_elbo(obs_seqs)

        for n_samples in range(1, args.num_samples, args.evolution_wrt_num_samples):

            offline_values, online_values = mc_elbos(obs_seqs, n_samples)[:-1]

            offline_values_list[n_samples] = offline_values
            online_values_list[n_samples] = online_values

        offline_values, online_values, (samples_seqs, weights_seqs, filt_state_seqs, backwd_state_seqs) = mc_elbos(obs_seqs, args.num_samples)

        offline_values_list[args.num_samples] = offline_values
        online_values_list[args.num_samples] = online_values
        
        errors_online = dict()
        errors_offline = dict()

        for k in offline_values_list.keys():
            errors_online_k = jnp.abs(true_values - online_values_list[k])
            errors_online[k] = {k:v.tolist() for k,v in enumerate(errors_online_k)}
            errors_offline_k  = jnp.abs(true_values - offline_values_list[k])
            errors_offline[k] = {k:v.tolist() for k,v in enumerate(errors_offline_k)}

        errors_online = pd.DataFrame(errors_online).apply(pd.Series).unstack().reset_index()
        errors_online['Method'] = 'Online'
        errors_offline = pd.DataFrame(errors_offline).apply(pd.Series).unstack().reset_index()   
        errors_offline['Method'] = 'Offline'
        errors = pd.concat([errors_online, errors_offline], axis=0).reset_index().drop(columns='index')

        errors.columns = ['Num samples', 'Seq nb', 'Value', 'Method']
        errors.to_csv(os.path.join(args.save_dir, 'errors_wrt_num_samples.csv'))
        sns.violinplot(data=errors, 
                x='Num samples', 
                y='Value',
                hue='Method',
                inner='point')
                
        plt.savefig(os.path.join(args.save_dir, 'errors_wrt_num_samples_violinplot.pdf'), format='pdf')
        plt.close()

    elif args.evolution_wrt_dim != -1: 
        key, *subkeys = jax.random.split(key, 3)

        true_values_list = dict()
        offline_values_list = dict()
        online_values_list = dict()

        for dim in range(args.evolution_wrt_dim, args.state_dim+1, args.evolution_wrt_dim):
            args_p.state_dim = args_q.state_dim = dim
            args_p.obs_dim = args_q.obs_dim = dim

            sampler, mc_elbos, true_elbo = get_sampler_and_elbos(subkeys[0], args, args_p, args_q)

            state_seqs, obs_seqs = sampler(subkeys[1])

            true_values = true_elbo(obs_seqs)
            offline_values, online_values = mc_elbos(obs_seqs, args.num_samples)[:-1]

            true_values_list[dim] = true_values
            offline_values_list[dim] = offline_values
            online_values_list[dim] = online_values


        true_values = true_elbo(obs_seqs)
        offline_values, online_values, (samples_seqs, weights_seqs, filt_state_seqs, backwd_state_seqs) = mc_elbos(obs_seqs, args.num_samples)

        true_values_list[args.state_dim] = true_values
        offline_values_list[args.state_dim] = offline_values
        online_values_list[args.state_dim] = online_values

        errors_online = dict()
        errors_offline = dict()

        for k in true_values_list.keys():
            errors_online_k = jnp.abs(true_values_list[k] - online_values_list[k])
            errors_online[k] = {k:v.tolist() for k,v in enumerate(errors_online_k)}
            errors_offline_k  = jnp.abs(true_values_list[k] - offline_values_list[k])
            errors_offline[k] = {k:v.tolist() for k,v in enumerate(errors_offline_k)}

        errors_online = pd.DataFrame(errors_online).apply(pd.Series).unstack().reset_index()
        errors_online['Method'] = 'Online'
        errors_offline = pd.DataFrame(errors_offline).apply(pd.Series).unstack().reset_index()   
        errors_offline['Method'] = 'Offline'
        errors = pd.concat([errors_online, errors_offline], axis=0).reset_index().drop(columns='index')

        errors.columns = ['State dim', 'Seq nb', 'Value', 'Method']

        errors.to_csv(os.path.join(args.save_dir, 'errors_wrt_state_dim.csv'))

        sns.violinplot(data=errors, 
                x='State dim', 
                y='Value',
                hue='Method',
                inner='point')

        # plt.ylabel('$\mathcal{L}(\\theta, \\lambda) - \hat{\mathcal{L}}(\\theta, \\lambda)$')

        plt.savefig(os.path.join(args.save_dir, 'errors_wrt_state_dim_violinplot.pdf'), format='pdf')
        plt.close()

    else: 
        key, *subkeys = jax.random.split(key, 3)

        sampler, mc_elbos, true_elbo = get_sampler_and_elbos(subkeys[0], args, args_p, args_q)

        true_values = true_elbo(obs_seqs)
        offline_values, online_values, (samples_seqs, weights_seqs, filt_state_seqs, backwd_state_seqs) = mc_elbos(obs_seqs, args.num_samples)




    #%%


def compute_distances_to_backward():

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

    fig, axes = plt.subplots(args.num_seqs, 1, figsize=(20,20))
    plt.tight_layout()
    plt.autoscale()
    for seq_nb in range(args.num_seqs):
        sns.violinplot(data=distances[seq_nb], x='Timestep',y='Value', ax=axes[seq_nb])
    plt.savefig(os.path.join(args.save_dir, 'kullback_distances.pdf'),format='pdf')
    plt.close()

# sns.violinplot(data=distances, x='Timestep', y='Value', hue='Sequence_nb')
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
    
    args.state_dim = 20
    args.obs_dim = 20
    args.num_samples = 200
    args.num_seqs = 20
    args.seq_length = 50
    args.trained_model = False
    args.range_transition_map_params = (0.99, 1)
    args.save_dir = 'experiments/tests/online/test_evolution'
    args.exp_dir = 'experiments/p_nonlinear/2022_07_27__12_21_27'
    args.method_name = 'johnson_freeze__theta'
    args.evolution_wrt_seq_length = -1
    args.evolution_wrt_num_samples = -1
    args.evolution_wrt_dim = 4


    if args.trained_model:

        args_p = utils.load_args('train_args',os.path.join(args.exp_dir, args.method_name))

        method_dir = os.path.join(args.exp_dir, args.method_name)
        args_q = utils.load_args('train_args', method_dir)
        args_p.linear = args_q.split('_')[0] == 'linear'

    else: 

        args_p = argparse.Namespace()
        args_p.state_dim, args_p.obs_dim = args.state_dim, args.obs_dim
        args_p.emission_bias = False
        args_p.p_version = 'linear'
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
        args_q.state_dim, args_q.obs_dim = args.state_dim, args.obs_dim
        args_q.q_version = 'linear'
        args_q.transition_matrix_conditionning = 'diagonal'
        args_q.range_transition_map_params = args.range_transition_map_params
        args_q.transition_bias = False
        args_q.emission_bias = False 
        args_q.update_layers = (8,)
        args_q.backwd_layers = False
        args_q.explicit_proposal = False 

    os.makedirs(args.save_dir, exist_ok=True)

    utils.save_args(args, 'args', args.save_dir)


    main(args, args_p, args_q)

# weights = jnp.exp(log_weights) / num_samples

# for t, weights_t in enumerate(weights):s
#     g = sns.displot(weights_t.flatten(), bins=100, kind='hist')
#     g.savefig(os.path.join(save_dir, f'{t}'))

