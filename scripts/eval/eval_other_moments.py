#%%
import jax, jax.numpy as jnp
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_platform_name', 'gpu')
jax.config.update('jax_enable_x64', False)
import dill

import pandas
import matplotlib.pyplot as plt
from src.variational import get_variational_model, NeuralBackwardSmoother
from src.stats.hmm import get_generative_model, LinearGaussianHMM
from src.stats.smc import SMC
from src.utils.misc import *
import os
import seaborn as sns 
model_name = 'johnson_backward,200.5.adam,1e-2,cst.reset,500,1.autodiff_on_backward.cpu.basic_logging'

key = jax.random.PRNGKey(0)
base_path = 'experiments/p_chaotic_rnn/offline_on_8_sequences_bis'
some_experiment_path =  os.path.join(base_path, os.listdir(base_path)[0])
p_args = load_args('args',some_experiment_path)
p_args.num_particles, p_args.num_smooth_particles = 10_000, 1_000
p = get_generative_model(p_args)
smc_engine = p.smc

some_variational_model_path = os.path.join(some_experiment_path, 
                                           os.listdir(some_experiment_path)[0])
q_args = load_args('args', some_variational_model_path)
q_args.state_dim, q_args.obs_dim = p_args.state_dim, p_args.obs_dim
q = get_variational_model(q_args)



def smc_smoothing_up_to_t(key, formatted_theta, y):
  key, key_filt = jax.random.split(key, 2)
  log_probs, particles = smc_engine.compute_filt_params_seq(key_filt, 
                                                            y, 
                                                            formatted_theta)[:-1]
  paths = []
  for t in tqdm(timesteps):
      key, key_smooth = jax.random.split(key, 2)
      paths.append(np.array(smc_engine.smoothing_paths_from_filt_seq(key_smooth,
                                                            (log_probs[:t], particles[:t]),
                                                            formatted_theta)))

  return paths

timesteps = range(50, p_args.seq_length+1, 50)



def load_seqs(base_path):
  xs, ys = [],[]
  for experiment_path in os.listdir(base_path):
    experiment_path = os.path.join(base_path, experiment_path)
    xs.append(jnp.load(os.path.join(experiment_path, 'state_seqs.npy'))[0])
    ys.append(jnp.load(os.path.join(experiment_path, 'obs_seqs.npy'))[0])
  return xs, ys

xs, ys = load_seqs(base_path)

def smc_paths_all_exps(base_path):
  smoothed_sequences_at_multiples_timesteps = []

  for y, experiment_path in zip(ys, os.listdir(base_path)):
    experiment_path = os.path.join(base_path, experiment_path)
    theta_star = load_params('theta_star', experiment_path)
    formatted_theta_star = p.format_params(theta_star)
    key, key_smc_smooth = jax.random.split(key, 2)
    smoothed_sequences_at_multiples_timesteps.append(smc_smoothing_up_to_t(
                                                      key_smc_smooth, 
                                                      formatted_theta_star, 
                                                      y))
    return smoothed_sequences_at_multiples_timesteps
  
def vi_marginals_all_exps(base_path):
  smoothed_sequences_at_multiple_timesteps = []
  for y, experiment_path in zip(ys, os.listdir(base_path)):
    experiment_path = os.path.join(base_path, experiment_path)
    variational_model_path = os.path.join(experiment_path, 
                                          os.listdir(experiment_path)[0])
    phi = load_params('phi', variational_model_path)
    smoothed_sequence_at_multiple_timesteps = []
    for t in tqdm(timesteps):
      smoothed_sequence_at_multiple_timesteps.append(q.smooth_seq(y[:t], phi))
    smoothed_sequences_at_multiple_timesteps.append(smoothed_sequence_at_multiple_timesteps)
  return smoothed_sequences_at_multiple_timesteps

def compute_backwd_params_sequences(base_path):
  backwd_params_all_sequences = []
  for y, experiment_path in zip(ys, os.listdir(base_path)):
    experiment_path = os.path.join(base_path, experiment_path)
    variational_model_path = os.path.join(experiment_path, 
                                          os.listdir(experiment_path)[0])
    formatted_phi = q.format_params(load_params('phi', variational_model_path))
    state_seq = q.compute_state_seq(y, len(y)-1, formatted_phi)
    backwd_params_seq = q.compute_backwd_params_seq(state_seq, formatted_phi)
    backwd_params_all_sequences.append(backwd_params_seq)
  return backwd_params_all_sequences

vi_backwd_params_all_seqs = compute_backwd_params_sequences(base_path)


def campbell_marginals_all_exps(base_path):
  smoothed_sequences = []
  for experiment_path in os.listdir(base_path):
    experiment_path = os.path.join(base_path, experiment_path)
    variational_model_path = load_args('args', experiment_path).load_from
    smoothed_sequence = jnp.load(os.path.join(variational_model_path, 'smoothing_stats.npy'))
    smoothed_sequences.append(smoothed_sequence)
  return smoothed_sequences

campbell_marginals = campbell_marginals_all_exps(base_path)

#%%
with open('smc_paths.dill', 'rb') as f: 
  # smc_paths = dill.dump(smc_paths, f)
  smc_paths = dill.load(f)


# vi_marginals = vi_marginals_all_exps(base_path)

with open('vi_marginals.dill', 'rb') as f: 
  # smc_paths = dill.dump(vi_marginals, f)
  vi_marginals = dill.load(f)

#%%
bad_exp_nbs = []
color = sns.color_palette()[1]
color_campbell = sns.color_palette()[2]
# alpha = 0.2

def errors_1st_moment(smc_paths, vi_marginals):
  
  additive_errors = {}
  marginal_errors = {}
  for exp_nb, (smc_path_all_t, vi_marginals_all_t) in enumerate(zip(smc_paths, vi_marginals)):
    if exp_nb not in bad_exp_nbs:
      additive_errors[exp_nb] = {}
      for t, smc_path_up_to_t, vi_marginals_up_to_t in zip(timesteps, smc_path_all_t, vi_marginals_all_t):
        smc_means = jnp.mean(smc_path_up_to_t, axis=0)
        vi_means = vi_marginals_up_to_t[0]
        additive_errors[exp_nb][t] = jnp.linalg.norm(jnp.sum(smc_means, axis=0) - jnp.sum(vi_means, axis=0)).tolist()
      marginal_errors_for_exp = jnp.linalg.norm(smc_means-vi_means, axis=1)
      marginal_errors[exp_nb] = {t:v.tolist() for t,v in enumerate(marginal_errors_for_exp) if t in timesteps}

  return additive_errors, marginal_errors

def errors_2nd_moment(smc_paths, vi_marginals):
  
  additive_errors = {}
  marginal_errors = {}
  for exp_nb, (smc_path_all_t, vi_marginals_all_t) in enumerate(zip(smc_paths, vi_marginals)):
    if exp_nb not in bad_exp_nbs:
      additive_errors[exp_nb] = {}
      for t, smc_path_up_to_t, vi_marginals_up_to_t in zip(timesteps, smc_path_all_t, vi_marginals_all_t):
        smc_2nd_moment = jnp.mean(smc_path_up_to_t**2, axis=0)
        vi_2nd_moment = vi_marginals_up_to_t[0]**2 + jax.vmap(jnp.diagonal)(vi_marginals_up_to_t[1])
        additive_errors[exp_nb][t] = jnp.linalg.norm(jnp.sum(smc_2nd_moment, axis=0) - jnp.sum(vi_2nd_moment, axis=0)).tolist()
      marginal_errors_for_exp = jnp.linalg.norm(smc_2nd_moment-vi_2nd_moment, axis=1)
      marginal_errors[exp_nb] = {t:v.tolist() for t,v in enumerate(marginal_errors_for_exp) if t in timesteps}

  return additive_errors, marginal_errors

def errors_crossprods(smc_paths, vi_marginals, vi_backwd_params):
  
  additive_errors = {}
  marginal_errors = {}

  def analytical_crossprod(vi_1st_moment, vi_2nd_moment, vi_backwd_params):
    return vi_backwd_params.map.w @ vi_2nd_moment + vi_backwd_params.map.b * vi_1st_moment
  
  for exp_nb, (smc_path_all_t, vi_marginals_all_t, vi_backwd_params_all_t) in enumerate(zip(smc_paths, vi_marginals, vi_backwd_params)):
    if exp_nb not in bad_exp_nbs:
      additive_errors[exp_nb] = {}
      for t, smc_path_up_to_t, vi_marginals_up_to_t in zip(timesteps, smc_path_all_t, vi_marginals_all_t):
        smc_crossprods = jnp.mean(smc_path_up_to_t[:,:-1]*smc_path_up_to_t[:,1:], axis=0)
        vi_1st_moment = vi_marginals_up_to_t[0]
        vi_2nd_moment = vi_1st_moment**2 + jax.vmap(jnp.diagonal)(vi_marginals_up_to_t[1])
        vi_backwd_params = tree_get_slice(0, t-1, vi_backwd_params_all_t)
        vi_cross_prods = jax.vmap(analytical_crossprod)(vi_1st_moment[1:], vi_2nd_moment[1:], vi_backwd_params)
        additive_errors[exp_nb][t] = jnp.linalg.norm(jnp.sum(smc_crossprods, axis=0) - jnp.sum(vi_cross_prods, axis=0)).tolist()
      marginal_errors_for_exp = jnp.linalg.norm(smc_crossprods-vi_cross_prods, axis=1)
      marginal_errors[exp_nb] = {t:v.tolist() for t,v in enumerate(marginal_errors_for_exp) if t in timesteps}

  return additive_errors, marginal_errors



def plot_marginal_errors(marginal_errors, ax):
  marginal_errors = pd.DataFrame.from_dict(marginal_errors).unstack().reset_index()
  marginal_errors.columns = ['Sequence', 'Timesteps', 'Marginal']
  
  sns.lineplot(marginal_errors, ax=ax, x='Timesteps', y='Marginal', style='Sequence', c=color, legend=False)



def plot_marginal_errors():
  pass

def marginal_errors_campbell(smc_paths, campbell_marginals):
  marginal_errors_1st_moment = {}
  marginal_errors_2nd_moment = {}
  marginal_errors_crossprods = {}

  for exp_nb in range(len(campbell_marginals)):
    smc_path_for_exp =  smc_paths[exp_nb][-1]
    campbell_path_for_exp = campbell_marginals[exp_nb]
    smc_1st_moment = jnp.mean(smc_path_for_exp, axis=0)
    campbell_1st_moment = jnp.mean(campbell_path_for_exp, axis=1)
    error_1st_moment = jnp.linalg.norm(smc_1st_moment - campbell_1st_moment, axis=1)

    smc_2nd_moment = jnp.mean(smc_path_for_exp**2, axis=0)
    campbell_2nd_moment = jnp.mean(campbell_path_for_exp**2, axis=1)
    error_2nd_moment = jnp.linalg.norm(smc_2nd_moment - campbell_2nd_moment, axis=1)

    smc_crossprods = jnp.mean(smc_path_for_exp[:,:-1]*smc_path_for_exp[:,1:], axis=0)
    campbell_crossprods = jnp.mean(campbell_path_for_exp[:-1]*campbell_path_for_exp[1:], axis=1)
    error_crossprods = jnp.linalg.norm(smc_crossprods - campbell_crossprods, axis=1)

    marginal_errors_1st_moment[exp_nb] = {t:v.tolist() for t,v in enumerate(error_1st_moment) if t in timesteps}
    marginal_errors_2nd_moment[exp_nb] = {t:v.tolist() for t,v in enumerate(error_2nd_moment) if t in timesteps}
    marginal_errors_crossprods[exp_nb] = {t:v.tolist() for t,v in enumerate(error_crossprods) if t in timesteps}

  return marginal_errors_1st_moment, marginal_errors_2nd_moment, marginal_errors_crossprods

def plot_marginals_campbell(x, x_sq, x1x2):
    fig, (ax0, ax1, ax2) = plt.subplots(1,3, figsize=(15,5))
    x = pd.DataFrame.from_dict(x).unstack().reset_index()
    x_sq = pd.DataFrame.from_dict(x_sq).unstack().reset_index()
    x1x2 = pd.DataFrame.from_dict(x1x2).unstack().reset_index()

    x.columns = ['Sequence', 'Timesteps', '1st moment']
    x_sq.columns = ['Sequence', 'Timesteps', '2nd moment']
    x1x2.columns = ['Sequence', 'Timesteps', 'Cross-products']
  
    # sns.lineplot(x, ax=ax0, x='Timesteps', y='1st moment', c=color_campbell)
    sns.lineplot(x, ax=ax0, x='Timesteps', y='1st moment', style='Sequence', c=color_campbell, legend=False)

    # sns.lineplot(x_sq, ax=ax1, x='Timesteps', y='2nd moment', c=color_campbell)
    sns.lineplot(x_sq, ax=ax1, x='Timesteps', y='2nd moment', style='Sequence', c=color_campbell, legend=False)

    # sns.lineplot(x1x2, ax=ax2, x='Timesteps', y='Cross-products', c=color_campbell)
    sns.lineplot(x1x2, ax=ax2, x='Timesteps', y='Cross-products', style='Sequence', c=color_campbell, legend=False)
    plt.tight_layout()
    plt.autoscale()
x_campbell, x_sq_campbell, x_cross_campbell = marginal_errors_campbell(smc_paths, campbell_marginals)
plot_marginals_campbell(x_campbell, x_sq_campbell, x_cross_campbell)
plt.savefig('campbell_marginals.png', format='png')
#%%

# additive_errors_1st_moment, marginal_errors_1st_moment = errors_1st_moment(smc_paths, vi_marginals)
# additive_errors_2nd_moment, marginal_errors_2nd_moment = errors_2nd_moment(smc_paths, vi_marginals)
# additive_errors_crossprods, marginal_errors_crossprods = errors_crossprods(smc_paths, vi_marginals,
#                                                                             vi_backwd_params_all_seqs)

# plot_errors(additive_errors_1st_moment, marginal_errors_1st_moment)
# plt.savefig('first_moments.png', format='png')
# plot_errors(additive_errors_2nd_moment, marginal_errors_2nd_moment)#%%
# plt.savefig('second_moments.png', format='png')
# plot_errors(additive_errors_crossprods, marginal_errors_crossprods)#%%
# plt.savefig('crossprods.png', format='png')

#%%
def additive_error_crossprods(smc_paths, vi_marginals):
  for smc_path_all_t, vi_marginals_all_t in zip(smc_paths, vi_marginals):
    additive_errors = []
    for smc_path_up_to_t, vi_marginals_up_to_t in zip(smc_path_all_t, vi_marginals_all_t):
      smc_cross_prods = jnp.mean(smc_path_up_to_t[:,:-1]*smc_path_up_to_t[:,1:], 
                                axis=0)
      vi_means = vi_marginals_up_to_t[0]
      vi_covs = jax.vmap(jnp.diagonal)(vi_marginals_up_to_t[1])
      vi_2nd_moment = vi_means ** 2 + vi_covs
      additive_errors.append(jnp.linalg.norm(jnp.sum(smc_cross_prods, axis=0) - jnp.sum(vi_2nd_moment, axis=0)))
    plt.plot(timesteps, additive_errors)
#%%
      

#%%


# @jax.jit


def variational_montecarlo_smoothing_up_to_t(key, y, timesteps):

  state_seq = q.compute_state_seq(y,   def _path_up_to_T(key, T):

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
    

# smc_paths = smc_smoothing_up_to_t(key, y, timesteps)



#%%
# jnp.save('smc_paths.npy', smc_paths)


# variational_paths = variational_montecarlo_smoothing_up_to_t(key, 
#                                                              y, 
#                                                              timesteps)

#%%
# svi_marginals = variational_analytical_marginals(y, timesteps)
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

# #%%
# def additive_errors_1st_moment(smc_paths, svi_marginals):
#   errors = []
#   for t, smc_path, svi_marginal in zip(timesteps, smc_paths, svi_marginals):
#     smc_1st_moments = jnp.mean(smc_path, axis=0)
#     svi_1st_moments = svi_marginal[0]
#     errors.append(jnp.linalg.norm(jnp.sum(smc_1st_moments - svi_1st_moments, axis=0)))
#   return errors

# def additive_errors_2nd_moment(smc_paths, svi_marginals):
#   errors = []
#   for smc_path, svi_marginal in zip(smc_paths, svi_marginals):
#     smc_2nd_moments = jnp.mean(smc_path**2, axis=0)
#     svi_2nd_moments = jax.vmap(jnp.diag)(svi_marginal[1]) + svi_marginal[0]**2
#     errors.append(jnp.linalg.norm(jnp.sum(smc_2nd_moments - svi_2nd_moments, axis=0)))
#   return errors

# errors_1st_moment = additive_errors_1st_moment(smc_paths, svi_marginals)
# # errors_2nd_moment = additive_errors_2nd_moment(smc_paths, svi_marginals)

# plt.plot(timesteps, errors_1st_moment)
# plt.plot(timesteps, errors_2nd_moment)

#%%



# errors_x, errors_x_squared, errors_x1x2 = additive_errors(smc_paths, variational_paths)

# plt.plot(timesteps, errors_x_squared, label='x^2')
# # plt.plot(timesteps, errors_x1x2, label='x1x2')
# plt.plot(timesteps, errors_x, label='x')
# plt.legend()

#%%

#%%

def plot_all_exps(smc_paths, vi_marginals):
  for exp_nb in range(len(os.listdir(base_path))):
    x_smoothed_smc = jnp.mean(smc_paths[exp_nb][-1], axis=0)
    x_smoothed_svi = vi_marginals[exp_nb][-1][0]
    x_true = xs[exp_nb]

    dims = p_args.state_dim
    fig, axes = plt.subplots(dims, 1, figsize=(15,1.5*p_args.state_dim))
    for d in range(dims):
        axes[d].plot(x_true[:,d], label='True')
        axes[d].legend()
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

