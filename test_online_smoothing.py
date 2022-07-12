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
num_seqs = 10
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
q = hmm.LinearGaussianHMM(state_dim, state_dim ,'diagonal', (0.5, 1), False, False)

key = jax.random.PRNGKey(0)

key, subkey_theta, subkey_phi = jax.random.split(key, 3)

theta = p.get_random_params(subkey_theta)
phi = q.get_random_params(subkey_phi)


# phi = theta
key, subkey = jax.random.split(key, 2)

state_seqs, obs_seqs = p.sample_multiple_sequences(subkey, theta, num_seqs, seq_length)

# normalizer = lambda x: jnp.mean(jnp.exp(x))
normalizer = smc.exp_and_normalize

closed_form_elbo = jax.vmap(jax.jit(lambda obs_seq: LinearGaussianELBO(p,q)(obs_seq, p.format_params(theta), q.format_params(phi))))
offline_mc_elbo = jax.vmap(jax.jit(lambda key, obs_seq: BackwardLinearELBO(p, q, num_samples)(key, obs_seq, p.format_params(theta), q.format_params(phi))))
# offline_mc_elbo = jax.vmap(jax.jit(lambda key, obs_seq: GeneralBackwardELBO(p, q, num_samples)(key, obs_seq, p.format_params(theta), q.format_params(phi))))

# online_mc_elbo = jax.vmap(jax.jit(lambda key, obs_seq: OnlineGeneralBackwardELBO(p, q, normalizer, num_samples)(key, obs_seq, p.format_params(theta), q.format_params(phi))))
online_mc_elbo = jax.vmap(jax.jit(lambda key, obs_seq: OnlineBackwardLinearELBO(p, q, normalizer, num_samples)(key, obs_seq, p.format_params(theta), q.format_params(phi))[0]))

keys = jax.random.split(key, num_seqs)
true_elbo_values = closed_form_elbo(obs_seqs)
offline_mc_elbo_paths = offline_mc_elbo(keys, obs_seqs)
online_mc_elbo_paths = online_mc_elbo(keys, obs_seqs)
import random
get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))


errors = jnp.array([[offline_mc_elbo_paths[seq_nb] - true_elbo_values[seq_nb], 
                    online_mc_elbo_paths[seq_nb] - true_elbo_values[seq_nb]] for seq_nb in range(num_seqs)])
fig, axes = plt.subplots(num_seqs, figsize=(20,50))
colors = get_colors(num_seqs)

for seq_nb in range(num_seqs): 
    sns.histplot(errors[seq_nb][0], ax=axes[seq_nb], color=colors[0])
    sns.histplot(errors[seq_nb][1], ax=axes[seq_nb], color=colors[1])

plt.legend()
plt.autoscale(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'histograms_offline_vs_online.pdf'),format='pdf')
# g = sns.FacetGrid(errors, row="smoker", col="time", margin_titles=True)

# sns.kdeplot(offline_errors, olor='red')
# sns.kdeplot(online_errors, color='blue')

#%%




# weights = jnp.exp(log_weights) / num_samples

# for t, weights_t in enumerate(weights):
#     g = sns.displot(weights_t.flatten(), bins=100, kind='hist')
#     g.savefig(os.path.join(save_dir, f'{t}'))

