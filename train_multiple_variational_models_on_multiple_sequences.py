#%%
import argparse 
from src.utils.misc import get_defaults, save_args, load_args, tree_get_slice
import os
from src.stats.hmm import get_generative_model
from src.variational import get_variational_model
from src.training import SVITrainer
import jax, jax.numpy as jnp
import dill 
import matplotlib
from functools import partial
from tqdm import tqdm
exp_path = 'experiments/eval_multiple_seqs/fitting_several_sequences'
load = True
os.makedirs(exp_path, exist_ok=True)

jax.config.update('jax_log_compiles', False)
jax.config.update('jax_platform_name', 'cpu')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

key = jax.random.PRNGKey(0)

def set_p_args():
  if load:
    p_args = load_args('p_args', exp_path)
  else: 
    p_args = argparse.Namespace()
    p_args.state_dim, p_args.obs_dim = 5,5
    p_args.model = 'chaotic_rnn'
    p_args.load_from = ''
    p_args.loaded_seq = False
    p_args.num_seqs = 50
    p_args.seq_length = 500
    p_args = get_defaults(p_args)
    save_args(p_args,'p_args',exp_path)
  p_args.num_particles = 10_000
  p_args.num_smooth_particles = 50
  return p_args

def set_q_args(p_args):
  if load: 
    q_args = load_args('q_args', exp_path)
  else:
    q_args = argparse.Namespace()
    q_args.state_dim, q_args.obs_dim = p_args.state_dim, p_args.obs_dim
    q_args.model = 'johnson_backward,100'
    q_args = get_defaults(q_args)
    q_args.optimizer = 'adam'
    q_args.learning_rate = 1e-2
    q_args.optim_options = 'cst'
    q_args.num_epochs = 1000
    q_args.num_samples = 2
    q_args.training_mode = f'reset,{p_args.seq_length},1'
    q_args.elbo_mode = 'autodiff_on_backward'
    q_args.logging_type = 'basic_logging'
    save_args(q_args, 'q_args', exp_path)
  return q_args

p_args = set_p_args()
q_args = set_q_args(p_args)

key, key_theta, key_seqs = jax.random.split(key, 3)
p, theta = get_generative_model(p_args, key_theta)
q = get_variational_model(q_args)

def get_sequences():
  
  if load: 
    xs = jnp.load(os.path.join(exp_path,'xs.npy'))
    ys = jnp.load(os.path.join(exp_path,'ys.npy'))
  else: 
    xs, ys = p.sample_multiple_sequences(
                                    key_seqs, 
                                    theta, 
                                    p_args.num_seqs, 
                                    p_args.seq_length, 
                                    single_split_seq=False,
                                    load_from=p_args.load_from,
                                    loaded_seq=p_args.loaded_seq)
    
  return xs, ys

xs, ys = get_sequences()

def get_params_q(key):
  if load:
    with open(os.path.join(exp_path, 'fitted_params'), 'rb') as f: 
      fitted_params = dill.load(f)
  else: 
    trainer = SVITrainer(
                    p, 
                    theta,
                    q,
                    q_args.optimizer,
                    q_args.learning_rate,
                    q_args.optim_options,
                    q_args.num_epochs, 
                    p_args.seq_length,
                    q_args.num_samples,
                    False,
                    '',
                    1,
                    q_args.training_mode,
                    q_args.elbo_mode,
                    q_args.logging_type)
    fitted_params = []
    final_elbos = []
    for seq_nb, (x,y) in enumerate(zip(xs,ys)):
      key, key_params, key_mc = jax.random.split(key, 3)
      data = (jnp.expand_dims(x,0), jnp.expand_dims(y,0))
      phi, final_elbo = trainer.fit(key_params, 
                                    key_mc, 
                                    data, 
                                    None,
                                    q_args,
                                    None)
      print(f'Final ELBO for sequence {seq_nb}:',final_elbo)
      fitted_params.append(phi)
      final_elbos.append(final_elbo)

  with open(os.path.join(exp_path, 'fitted_params'), 'wb') as f: 
      dill.dump(fitted_params, f)

  return fitted_params

key, subkey = jax.random.split(key, 2)
params_q = get_params_q(subkey)
smc_engine = p.smc
timesteps = range(50,p_args.seq_length+1,50)
bad_exp_nbs = []

def smc_path_at_multiple_timesteps(key, ys):
  formatted_theta = p.format_params(theta)
  paths = []

  for y in tqdm(ys): 
    key, subkey_filt = jax.random.split(key, 2)

    paths_for_y = []
    log_probs, particles = smc_engine.compute_filt_params_seq(subkey_filt, y, formatted_theta)[:-1]
    for t in timesteps:
      filt_seqs = log_probs[:t], particles[:t]
      key, subkey_smooth = jax.random.split(key, 2)
      paths_for_y.append(smc_engine.smoothing_paths_from_filt_seq(subkey_smooth,
                                                        filt_seqs,
                                                        formatted_theta))
    paths.append(paths_for_y)

  return paths
        
def get_smc_paths(key, load):
  if load: 
    with open(os.path.join(exp_path, 'smc_paths.dill'), 'rb') as f: 
      paths = dill.load(f)
  else:
    paths = smc_path_at_multiple_timesteps(key, ys)
    with open(os.path.join(exp_path, 'smc_paths.dill'), 'wb') as f:
      dill.dump(paths, f)
  return paths

def vi_marginals_at_multiple_timesteps(params_q):
  vi_marginals = []
  for phi, y in tqdm(zip(params_q, ys)):
    vi_marginals_for_y = []
    for t in timesteps:
      vi_marginals_for_y.append(q.smooth_seq(y[:t], phi))
    vi_marginals.append(vi_marginals_for_y)
  return vi_marginals

def get_vi_backwd_params(params_q):
  backwd_params_all_sequences = []
  for phi, y in tqdm(zip(params_q, ys)): 
    formatted_phi = q.format_params(phi)
    state_seq = q.compute_state_seq(y, len(y)-1, formatted_phi)
    backwd_params_seq = q.compute_backwd_params_seq(state_seq, formatted_phi)
    backwd_params_all_sequences.append(backwd_params_seq)
  return backwd_params_all_sequences

def get_vi_marginals(params_q, load):
  if load:
    with open(os.path.join(exp_path,'vi_marginals.dill'), 'rb') as f: 
      vi_marginals = dill.load(f)
  else: 
    vi_marginals = vi_marginals_at_multiple_timesteps(params_q)
    with open(os.path.join(exp_path,'vi_marginals.dill'), 'wb') as f: 
      dill.dump(vi_marginals, f)
  return vi_marginals


smc_paths = get_smc_paths(key, load)
vi_marginals = get_vi_marginals(params_q, load)
vi_backwd_params = get_vi_backwd_params(params_q)

#%%
def errors_1st_moment(smc_paths, vi_marginals):
  
  additive_errors = {}
  marginal_errors = {}
  for exp_nb, (smc_path_all_t, vi_marginals_all_t) in enumerate(zip(smc_paths, vi_marginals)):
    if exp_nb not in bad_exp_nbs:
      additive_errors[exp_nb] = {}
      for t, smc_path_up_to_t, vi_marginals_up_to_t in zip(timesteps, smc_path_all_t, vi_marginals_all_t):
        smc_means = jnp.mean(jax.vmap(jax.vmap(partial(jnp.linalg.norm,ord=1)))(smc_path_up_to_t), axis=0)
        vi_means = jnp.linalg.norm(vi_marginals_up_to_t[0], axis=1, ord=1)
        additive_errors[exp_nb][t] = jnp.abs(jnp.sum(smc_means, axis=0) \
                                                     - jnp.sum(vi_means, axis=0)).tolist() / p.state_dim
      marginal_errors_for_exp = jnp.abs(smc_means - vi_means) / p.state_dim
      marginal_errors[exp_nb] = {t:v.tolist() for t,v in enumerate(marginal_errors_for_exp) if t in timesteps}

  return additive_errors, marginal_errors

def errors_2nd_moment(smc_paths, vi_marginals):
  
  additive_errors = {}
  marginal_errors = {}
  for exp_nb, (smc_path_all_t, vi_marginals_all_t) in enumerate(zip(smc_paths, vi_marginals)):
    if exp_nb not in bad_exp_nbs:
      additive_errors[exp_nb] = {}
      for t, smc_path_up_to_t, vi_marginals_up_to_t in zip(timesteps, smc_path_all_t, vi_marginals_all_t):
        # smc_2nd_moment = jnp.mean(smc_path_up_to_t**2, axis=0)
        # vi_2nd_moment = vi_marginals_up_to_t[0]**2 + jax.vmap(jnp.diagonal)(vi_marginals_up_to_t[1])
        # additive_errors[exp_nb][t] = jnp.linalg.norm(jnp.sum(smc_2nd_moment, axis=0) \
        #                                              - jnp.sum(vi_2nd_moment, axis=0), ord=1).tolist() / p.state_dim
        smc_2nd_moment = jnp.mean(jax.vmap(jax.vmap(lambda x:x.T @ x))(smc_path_up_to_t), axis=0)
        vi_2nd_moment = jnp.sum(vi_marginals_up_to_t[0]**2 + jax.vmap(jnp.diagonal)(vi_marginals_up_to_t[1]), 
                                axis=1)
        additive_errors[exp_nb][t] = jnp.abs(jnp.sum(smc_2nd_moment, axis=0) \
                                                     - jnp.sum(vi_2nd_moment, axis=0)).tolist() / p.state_dim
      marginal_errors_for_exp = jnp.abs(smc_2nd_moment-vi_2nd_moment) / p.state_dim
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
        smc_crossprods = jnp.mean(jax.vmap(jax.vmap(lambda x,y: x.T @ y))(smc_path_up_to_t[:,:-1], smc_path_up_to_t[:,1:]), axis=0)
        vi_1st_moment = vi_marginals_up_to_t[0]
        vi_2nd_moment = vi_1st_moment**2 + jax.vmap(jnp.diagonal)(vi_marginals_up_to_t[1])
        vi_backwd_params = tree_get_slice(0, t-1, vi_backwd_params_all_t)
        vi_cross_prods = jnp.sum(jax.vmap(analytical_crossprod)(vi_1st_moment[1:], vi_2nd_moment[1:], vi_backwd_params), axis=1)
        additive_errors[exp_nb][t] = jnp.abs(jnp.sum(smc_crossprods, axis=0) - jnp.sum(vi_cross_prods, axis=0)).tolist() / p.state_dim
      marginal_errors_for_exp = jnp.abs(smc_crossprods-vi_cross_prods) / p.state_dim
      marginal_errors[exp_nb] = {t:v.tolist() for t,v in enumerate(marginal_errors_for_exp) if t in timesteps}

  return additive_errors, marginal_errors

additive_errors_1st_moment, marginal_errors_1st_moment = errors_1st_moment(smc_paths, vi_marginals)
additive_errors_2nd_moment, marginal_errors_2nd_moment = errors_2nd_moment(smc_paths, vi_marginals)
additive_errors_crossprods, marginal_errors_crossprods = errors_crossprods(smc_paths, vi_marginals, vi_backwd_params)

#%%
color = sns.color_palette()[1]
alpha = 0.2
def plot_errors(errors, name, ax):
  errors = pd.DataFrame.from_dict(errors).unstack().reset_index()
  errors.columns = ['Sequence', 'Timesteps', name]
  sns.lineplot(errors, ax=ax, x='Timesteps', y=name, c=color)
  sns.lineplot(errors, ax=ax, x='Timesteps', y=name, style='Sequence', c=color, legend=False, alpha=alpha)  


matplotlib.rc('font',size=15)
# matplotlib.rcParams['font.family'] = 'STIXGeneral'
# matplotlib.rcParams['mathtext.fontset'] = 'STIX'

fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(2,3,figsize=(15,10))

plot_errors(additive_errors_1st_moment, 'Additive error against FFBSi', ax0) 
plot_errors(additive_errors_2nd_moment, 'Additive error against FFBSi', ax1) 
plot_errors(additive_errors_crossprods, 'Additive error against FFBSi', ax2) 

plot_errors(marginal_errors_1st_moment, 'Marginal error against FFBSi', ax3) 
plot_errors(marginal_errors_2nd_moment, 'Marginal error against FFBSi', ax4) 
plot_errors(marginal_errors_crossprods, 'Marginal error against FFBSi', ax5) 
plt.tight_layout()
plt.autoscale(True)
plt.savefig('errors_on_other_moments.png')
#%%


