#%%
import argparse
import haiku as hk 
import jax 
import jax.numpy as jnp
import backward_ica.hmm as hmm 
import backward_ica.utils as utils 
import backward_ica.smc as smc
from backward_ica.svi import BackwardLinearELBO, GeneralBackwardELBO, LinearGaussianELBO, OnlineBackwardLinearELBO, OnlineGeneralBackwardELBO
import seaborn as sns
import os 
import pandas as pd
import matplotlib.pyplot as plt
import random

num_seqs = 5
num_samples = 1000
num_indices = 5
seq_length = 50
num_particles = 1000
num_smooth_particles = 1000

save_dir = 'experiments/tests/online/trained_nonlinear_model'
os.makedirs(save_dir, exist_ok=True)


exp_dir = 'experiments/p_nonlinear/2022_07_27__12_21_27'

method_name = 'johnson_freeze__theta'

train_args = utils.load_args('train_args',os.path.join(exp_dir, method_name))
utils.set_global_cov_mode(train_args)


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


theta_star = utils.load_params('theta', os.path.join(exp_dir, method_name))


method_dir = os.path.join(exp_dir, method_name)
args = utils.load_args('train_args', method_dir)

if 'linear' in args.q_version:

    q = hmm.LinearGaussianHMM(state_dim=args.state_dim, 
                            obs_dim=args.obs_dim, 
                            transition_matrix_conditionning=args.transition_matrix_conditionning,
                            range_transition_map_params=args.range_transition_map_params,
                            transition_bias=args.transition_bias,
                            emission_bias=args.emission_bias) 

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

key = jax.random.PRNGKey(0)


phi = utils.load_params('phi', method_dir)[1]

# phi = theta_star
key, subkey = jax.random.split(key, 2)

state_seqs, obs_seqs = p.sample_multiple_sequences(subkey, theta_star, num_seqs, seq_length)

# normalizer = lambda x: jnp.mean(jnp.exp(x))
normalizer = smc.exp_and_normalize

# closed_form_elbo = jax.jit(jax.vmap(lambda obs_seq: LinearGaussianELBO(p,q)(obs_seq, p.format_params(theta_star), q.format_params(phi))))
# offline_mc_elbo = jax.vmap(jax.jit(lambda key, obs_seq: BackwardLinearELBO(p, q, num_samples)(key, obs_seq, p.format_params(theta_star), q.format_params(phi))))
offline_mc_elbo = jax.jit(jax.vmap(lambda key, obs_seq: GeneralBackwardELBO(p, q, num_samples)(key, obs_seq, p.format_params(theta_star), q.format_params(phi))))

online_mc_elbo = jax.jit(jax.vmap(lambda key, obs_seq: OnlineGeneralBackwardELBO(p, q, normalizer, num_samples)(key, obs_seq, p.format_params(theta_star), q.format_params(phi))))
# online_mc_elbo = jax.vmap(jax.jit(lambda key, obs_seq: OnlineBackwardLinearELBO(p, q, normalizer, num_samples)(key, obs_seq, p.format_params(theta_star), q.format_params(phi))[0]))

keys = jax.random.split(key, num_seqs)
# true_elbo_values = closed_form_elbo(obs_seqs)
offline_mc_elbo_values = offline_mc_elbo(keys, obs_seqs)
online_mc_elbo_values, (samples_seqs, weights_seqs, backwd_state_seqs) = online_mc_elbo(keys, obs_seqs)


state_dim = train_args.state_dim
get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
colors = get_colors(num_indices)

for seq_nb in range(num_seqs):
    
    samples_seq = samples_seqs[seq_nb]
    weights_seq = weights_seqs[seq_nb]
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
        # plt.savefig('')


# get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
# g = sns.FacetGrid(errors, row="smoker", col="time", margin_titles=True)

# sns.kdeplot(offline_errors, olor='red')
# sns.kdeplot(online_errors, color='blue')

#%%



# weights = jnp.exp(log_weights) / num_samples

# for t, weights_t in enumerate(weights):s
#     g = sns.displot(weights_t.flatten(), bins=100, kind='hist')
#     g.savefig(os.path.join(save_dir, f'{t}'))

