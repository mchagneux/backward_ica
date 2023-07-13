#%%
import jax, jax.numpy as jnp
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', False)
import dill

import pandas
import matplotlib.pyplot as plt
from src.variational import get_variational_model, NeuralBackwardSmoother
from src.stats.hmm import get_generative_model, LinearGaussianHMM
from src.stats.smc import SMC
from src.utils.misc import *
import os


key = jax.random.PRNGKey(0)
experiment_path = 'experiments/p_chaotic_rnn/2023_07_11__16_34_50'
model_path = 'johnson_backward,200.5.adam,1e-2,cst.reset,500,1.autodiff_on_backward.cpu.basic_logging'
full_model_path = os.path.join(experiment_path, model_path)
p_args = load_args('args', experiment_path)
p_args.num_particles, p_args.num_smooth_particles = 1e6, 100_000

x_true = jnp.load(os.path.join(experiment_path, 'state_seqs.npy'))[0]
y = jnp.load(os.path.join(experiment_path, 'obs_seqs.npy'))[0]


p = get_generative_model(p_args)
theta_star = load_params('theta_star', experiment_path)
formatted_theta_star = p.format_params(theta_star)

q_args = load_args('args', full_model_path)
q_args.state_dim, q_args.obs_dim = p_args.state_dim, p_args.obs_dim
q = get_variational_model(q_args, p)
phi = load_params('phi', full_model_path)
formatted_phi = q.format_params(phi)

key, key_smc_smooth = jax.random.split(key, 2)
smc_engine = p.smc


# @jax.jit
def smc_smoothing_up_to_t(key, y, timesteps):
  key, key_filt = jax.random.split(key, 2)
  log_probs, particles = smc_engine.compute_filt_params_seq(key_filt, 
                                                            y, 
                                                            formatted_theta_star)[:-1]
  paths = []
  for t in tqdm(timesteps):
      key, key_smooth = jax.random.split(key, 2)
      paths.append(np.array(smc_engine.smoothing_paths_from_filt_seq(key_smooth,
                                                            (log_probs[:t], particles[:t]),
                                                            formatted_theta_star)))

  return paths

def variational_montecarlo_smoothing_up_to_t(key, y, timesteps):

  state_seq = q.compute_state_seq(y, 
                                  len(y)-1, 
                                  formatted_phi)

  def _path_up_to_T(key, T):

    state_seq_up_to_T = tree_get_slice(0, T, state_seq)
    
    masks = jnp.arange(0, T) < T - 1
    # print(masks)

    def _backward_trajectory(key):

      def _sample(carry, x):
        mask, key, state_t = x
        x_tp1, state_tp1  = carry
        def _terminal_sample(key, x_tp1, states):
          filt_params = q.filt_params_from_state(states[0], formatted_phi)
          return q.filt_dist.sample(key, filt_params)
        def _backwd_sample(key, x_tp1, states):
          backwd_params = q.backwd_params_from_states(states, formatted_phi)
          return q.backwd_kernel.sample(key, x_tp1, backwd_params)
        
        sample = jax.lax.cond(mask, _backwd_sample, _terminal_sample, 
                            key, x_tp1, (state_t, state_tp1))
        return (sample, state_t), sample
      

      return jax.lax.scan(_sample, 
                          init=(jnp.empty((p.state_dim,)), 
                                tree_get_idx(-1, state_seq_up_to_T)),
                          xs=(masks, 
                              jax.random.split(key, T), 
                              state_seq_up_to_T))[1]
    
    return jax.vmap(_backward_trajectory)(jax.random.split(key, 
                                                           10_000))
  
  paths = []
  for t in tqdm(timesteps):
    key, subkey = jax.random.split(key, 2)
    paths.append(np.array(_path_up_to_T(subkey, t)))

  return paths
    
def variational_analytical_marginals(y, timesteps):
  marginals = []
  for t in tqdm(timesteps): 
    marginals.append(q.smooth_seq(y[:t], phi))

  return marginals


timesteps = jnp.arange(0, len(y)+1, 10)[1:]

# smc_paths = smc_smoothing_up_to_t(key, y, timesteps)
with open('smc_paths.dill', 'rb') as f: 
  smc_paths = dill.load(f)
  # smc_paths = dill.load(f)



#%%
# jnp.save('smc_paths.npy', smc_paths)


# variational_paths = variational_montecarlo_smoothing_up_to_t(key, 
#                                                              y, 
#                                                              timesteps)

#%%
svi_marginals = variational_analytical_marginals(y, timesteps)
#%%


#%%

# def x_on_path(path):
#   return jnp.sum(jnp.mean(path, axis=0), axis=0)

# def x_squared_on_path(path):
#   return jnp.sum(jnp.mean(path**2, axis=0), axis=0)

# def x1x2_on_path(path):
#   path = jnp.transpose(path, axes=(1,0,2))
#   cross_products = jax.vmap(lambda x,y: x*y)(path[:-1], 
#                                              path[1:])
#   return jnp.mean(cross_products, axis=1)

#%%
def additive_errors_1st_moment(smc_paths, svi_marginals):
  errors = []
  for t, smc_path, svi_marginal in zip(timesteps, smc_paths, svi_marginals):
    smc_1st_moments = jnp.mean(smc_path, axis=0)
    svi_1st_moments = svi_marginal[0]
    errors.append(jnp.linalg.norm(jnp.sum(smc_1st_moments - svi_1st_moments, axis=0)))
  return errors

def additive_errors_2nd_moment(smc_paths, svi_marginals):
  errors = []
  for smc_path, svi_marginal in zip(smc_paths, svi_marginals):
    smc_2nd_moments = jnp.mean(smc_path**2, axis=0)
    svi_2nd_moments = jax.vmap(jnp.diag)(svi_marginal[1]) + svi_marginal[0]**2
    errors.append(jnp.linalg.norm(jnp.sum(smc_2nd_moments - svi_2nd_moments, axis=0)))
  return errors

errors_1st_moment = additive_errors_1st_moment(smc_paths, svi_marginals)
errors_2nd_moment = additive_errors_2nd_moment(smc_paths, svi_marginals)

plt.plot(timesteps, errors_1st_moment)
# plt.plot(timesteps, errors_2nd_moment)

#%%



# errors_x, errors_x_squared, errors_x1x2 = additive_errors(smc_paths, variational_paths)

# plt.plot(timesteps, errors_x_squared, label='x^2')
# # plt.plot(timesteps, errors_x1x2, label='x1x2')
# plt.plot(timesteps, errors_x, label='x')
# plt.legend()

#%%

#%%

x_smoothed_smc = jnp.mean(smc_paths[-1], axis=0)
# x_smoothed_svi = jnp.mean(variational_paths[-1], axis=0)
x_smoothed_svi = svi_marginals[-1][0]
# x_smoothed_svi = q.smooth_seq(y, phi)[0]
# x_smoothed_svi = q.smooth_seq(y, phi)[0]

dims = p_args.state_dim
fig, axes = plt.subplots(dims, 1, figsize=(15,1.5*p_args.state_dim))
for d in range(dims):
    # axes[d].plot(x_true[:,d], label='True')
    # axes[d].legend()
    axes[d].plot(x_smoothed_smc[:,d], label='FFBSi')
    axes[d].legend()
    axes[d].plot(x_smoothed_svi[:,d], label='Variational')
    axes[d].legend()
#%%

# seq_length = p_args.seq_length
# q_args = load_args('args', full_model_path)
# q_args.state_dim, q_args.obs_dim = p_args.state_dim, p_args.obs_dim
# q = get_variational_model(q_args, p)
# phi = load_params('phi', full_model_path)

