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

state_dim = 10
obs_dim = 10
num_seqs = 5
num_samples = 1000

seq_length = 50
save_dir = 'experiments/tests/online'

args = argparse.Namespace()

args.parametrization = 'cov_chol'
import math
args.default_prior_mean = 0.0
args.range_transition_map_params = [0.99,1]
args.default_prior_base_scale = math.sqrt(1e-2)
args.default_transition_base_scale = math.sqrt(1e-2)
args.default_emission_base_scale = math.sqrt(1e-3)
args.default_transition_bias = 0
args.transition_bias = False
utils.set_global_cov_mode(args)

p = hmm.LinearGaussianHMM(state_dim, obs_dim, 'diagonal', (0.99,1), False, False)
q = hmm.LinearGaussianHMM(state_dim, obs_dim ,'diagonal', (0.8, 1), False, False)

key = jax.random.PRNGKey(0)

key, subkey_theta, subkey_phi = jax.random.split(key, 3)

theta = p.get_random_params(subkey_theta)
phi = q.get_random_params(subkey_phi)
# phi = theta

# phi = theta
key, subkey = jax.random.split(key, 2)

state_seqs, obs_seqs = p.sample_multiple_sequences(subkey, theta, num_seqs, seq_length)

# normalizer = lambda x: jnp.mean(jnp.exp(x))
normalizer = smc.exp_and_normalize

closed_form_elbo = jax.vmap(jax.jit(lambda obs_seq: LinearGaussianELBO(p,q)(obs_seq, p.format_params(theta), q.format_params(phi))))
# offline_mc_elbo = jax.vmap(jax.jit(lambda key, obs_seq: BackwardLinearELBO(p, q, num_samples)(key, obs_seq, p.format_params(theta), q.format_params(phi))))
offline_mc_elbo = jax.vmap(jax.jit(lambda key, obs_seq: GeneralBackwardELBO(p, q, num_samples)(key, obs_seq, p.format_params(theta), q.format_params(phi))))

online_mc_elbo = jax.vmap(jax.jit(lambda key, obs_seq: OnlineGeneralBackwardELBO(p, q, normalizer, num_samples)(key, obs_seq, p.format_params(theta), q.format_params(phi))))
# online_mc_elbo = jax.vmap(jax.jit(lambda key, obs_seq: OnlineBackwardLinearELBO(p, q, normalizer, num_samples)(key, obs_seq, p.format_params(theta), q.format_params(phi))[0]))

keys = jax.random.split(key, num_seqs)
true_elbo_values = closed_form_elbo(obs_seqs)
offline_mc_elbo_values = offline_mc_elbo(keys, obs_seqs)
online_mc_elbo_values, (samples_seqs, weights_seqs, backwd_state_seqs) = online_mc_elbo(keys, obs_seqs)


import random

get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
num_indices = 5
colors = get_colors(num_indices)
import numpy as np

for seq_nb in range(num_seqs):
    
    samples_seq = samples_seqs[seq_nb]
    weights_seq = weights_seqs[seq_nb]
    backwd_state_seq = utils.tree_get_idx(seq_nb, backwd_state_seqs)

    for time_idx in range(0, seq_length, seq_length // 5):
        fig, axes = plt.subplots(state_dim+1, 1, figsize=(20,30))

        samples = samples_seq[time_idx]
        key, subkey = jax.random.split(key, 2)

        random_indices = jax.random.choice(subkey, 
                                        jnp.arange(0, len(samples_seq[time_idx+1])), 
                                        shape=(num_indices,),
                                        replace=False)
        for dim_nb in range(state_dim):

            sns.histplot(samples[:,dim_nb], 
                        ax=axes[dim_nb], 
                        stat='density',
                        label=f'$\\xi_t^j[{dim_nb}]$',
                        color='grey')

        for num_idx in range(num_indices):
            random_idx = random_indices[num_idx]



            next_sample = samples_seq[time_idx+1][random_idx]
            weights = weights_seq[time_idx][random_idx]

            backwd_params = q.backwd_kernel.map(next_sample, utils.tree_get_idx(time_idx, backwd_state_seq))

            for dim_nb in range(state_dim):
                samples_x = samples[:,dim_nb]
                range_x = jnp.linspace(samples_x.min(), samples_x.max(), 100)
                mu, sigma = backwd_params.mean[dim_nb], backwd_params.scale.cov[dim_nb, dim_nb]
                backwd_pdf = lambda x: hmm.gaussian_pdf(x, mu, sigma)
                                                        
                axes[dim_nb].plot(range_x, backwd_pdf(range_x), label=f'$q(x_t[{dim_nb}] | \\xi_{{t+1}}^{{{random_idx}}})$', color=colors[num_idx])
                axes[dim_nb].legend()
            sns.histplot(weights, ax=axes[state_dim], label=f'$\\omega_t^{{{random_idx}}}j$', color=colors[num_idx])
            axes[state_dim].legend()


        plt.suptitle(f'Sequence {seq_nb}, time {time_idx}, (online/offline ELBO error {jnp.abs(true_elbo_values[seq_nb] - online_mc_elbo_values[seq_nb]):.2f}/{jnp.abs(true_elbo_values[seq_nb] - offline_mc_elbo_values[seq_nb]):.2f})')
        plt.autoscale(True)
        plt.tight_layout()

        # sns.pairplot(data=samples, 
        #                 diag_kws={'weights':weights}, 
        #                 plot_kws={'weights':weights}, kind="kde")
        plt.savefig(os.path.join('experiments','tests', 'online', f'seq_{seq_nb}_time_{time_idx}'))
        plt.close()
        # plt.savefig('')


# get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
# g = sns.FacetGrid(errors, row="smoker", col="time", margin_titles=True)

# sns.kdeplot(offline_errors, olor='red')
# sns.kdeplot(online_errors, color='blue')

#%%



# weights = jnp.exp(log_weights) / num_samples

# for t, weights_t in enumerate(weights):
#     g = sns.displot(weights_t.flatten(), bins=100, kind='hist')
#     g.savefig(os.path.join(save_dir, f'{t}'))

