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
import matplotlib.pyplot as plt

state_dim = 2
obs_dim = 2
num_seqs = 1
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
online_mc_elbo_values, (samples_seqs, log_probs_seqs, backwd_state_seqs) = online_mc_elbo(keys, obs_seqs)

print('Offline ELBO error:', jnp.mean(jnp.abs(true_elbo_values - offline_mc_elbo_values)))
print('Online ELBO error:', jnp.mean(jnp.abs(true_elbo_values - online_mc_elbo_values)))
n_pts = 1000
import random

get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
colors = get_colors(3)
for seq_nb in range(num_seqs):
    samples_seq = samples_seqs[seq_nb]
    log_probs_seq = log_probs_seqs[seq_nb]
    backwd_state_seq = utils.tree_get_idx(seq_nb, backwd_state_seqs)
    for time_idx in range(0, seq_length, seq_length // 10):
        fig, ax = plt.subplots(1,1)
        samples = samples_seq[time_idx]
        weights = smc.exp_and_normalize(log_probs_seq[time_idx])
        x_min, x_max = samples[:,0].min()-1, samples[:,0].max()+1
        y_min, y_max = samples[:,1].min()-1, samples[:,1].max()+1
        x, y = jnp.meshgrid(jnp.linspace(x_min, x_max, n_pts), 
                            jnp.linspace(y_min, y_max, n_pts))
        pos = jnp.dstack((x,y))
        key, subkey = jax.random.split(key, 2)
        next_sample = jax.random.choice(subkey, samples_seq[time_idx+1])
        backwd_params = q.backwd_kernel.map(next_sample, utils.tree_get_idx(time_idx, backwd_state_seq))
        ax = utils.confidence_ellipse(backwd_params.mean, backwd_params.scale.cov, ax=ax, c='black',n_std=1.96)
        sns.kdeplot(weights=weights, fill=True, x=samples[:,0], y=samples[:,1], ax=ax)
        plt.savefig(os.path.join('experiments','tests', 'online', f'seq_{seq_nb}_time_{time_idx}'))
        plt.clf()


# get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
# g = sns.FacetGrid(errors, row="smoker", col="time", margin_titles=True)

# sns.kdeplot(offline_errors, olor='red')
# sns.kdeplot(online_errors, color='blue')

#%%



# weights = jnp.exp(log_weights) / num_samples

# for t, weights_t in enumerate(weights):
#     g = sns.displot(weights_t.flatten(), bins=100, kind='hist')
#     g.savefig(os.path.join(save_dir, f'{t}'))

