import pandas as pd 
import seaborn as sns 
import numpy as np 
import dill 
import os 
import matplotlib.pyplot as plt
from datetime import datetime
from backward_ica.utils import save_args, load_args
import argparse 
import jax.numpy as jnp
import jax
exp_type = 'Sequence'

exp_dirs = ['experiments/p_linear_transition_with_nonlinear_emission/2022_10_31__14_14_47']

exp_names = ['All subsequences', 'Whole sequence only']
            
ref = 'smc'

date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

eval_dir = os.path.join('experiments', 'combine_evals', date)
os.makedirs(eval_dir, exist_ok=True)

evals_additive = dict()
evals_marginals = dict()
method_names = [
        "johnson_backward",
        "neural_backward_linear",
        "johnson_forward"
    ]

args = argparse.Namespace()
args.exp_dirs = exp_dirs 
args.method_names = method_names 
args.eval_dir = eval_dir

save_args(args, 'args', eval_dir)
up_to = None

def compute_errors(means_ref, means_q, slices):
                
    marginals = jnp.linalg.norm((means_q[-1] - means_ref[-1]), ord=1, axis=1)[slices]
    
    additive = []
    for means_ref_n, means_q_n in zip(means_ref, means_q):
        additive.append(jnp.linalg.norm(jnp.sum(means_ref_n - means_q_n, axis=0), ord=1))
    additive = jnp.array(additive)
    return marginals, additive

if exp_type != 'Sequence':
    for exp_name, exp_dir in zip(exp_names, exp_dirs):

        evals_marginals[exp_name] = dict()
        evals_additive[exp_name] = dict()

        for method_name in method_names: 
            method_eval_dir = os.path.join(exp_dir, 'evals', method_name)
            if not os.path.exists(method_eval_dir):
                continue 
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
                pretty_name = f"Linear {method_name.split('_')[1]}"


            with open(os.path.join(method_eval_dir, f'eval.dill'), 'rb') as f: 
                means_q, means_ref, slices = dill.load(f)
                marginals, additive = jax.vmap(compute_errors, in_axes=(0,0,None))(means_ref, means_q, slices)
                marginals /= means_q[0].shape[-1]
                additive /= means_q[0].shape[-1]
                evals_marginals[exp_name][pretty_name] = marginals.squeeze().tolist()
                evals_additive[exp_name][pretty_name] = additive.squeeze().tolist()

else: 
    num_seqs = load_args('args', os.path.join(exp_dirs[0], 'evals', method_names[0])).num_seqs

    for seq_nb in range(num_seqs):

        evals_marginals[seq_nb] = dict()
        evals_additive[seq_nb] = dict()

        for method_name in method_names: 

            method_eval_dir = os.path.join(exp_dirs[0], 'evals', method_name)
            if not os.path.exists(method_eval_dir):
                continue 
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
                pretty_name = f"Linear {method_name.split('_')[1]}"

            with open(os.path.join(method_eval_dir, f'eval.dill'), 'rb') as f: 
                means_q, means_ref, slices = dill.load(f)
                if ref == 'smc':
                    with open(os.path.join(os.path.join(exp_dirs[0], 'evals', 'ffbsi'), f'eval.dill'), 'rb') as f: 
                        means_ref, _ , _ = dill.load(f)
                elif ref == 'linear_0':
                    with open(os.path.join(os.path.join(exp_dirs[0], 'evals', 'linear_0'), f'eval.dill'), 'rb') as f: 
                        means_ref, _ , _ = dill.load(f)
                marginals, additive = jax.vmap(compute_errors, in_axes=(0,0,None))(means_ref, means_q, slices)
                marginals /= means_q[0].shape[-1]
                additive /= means_q[0].shape[-1]
                evals_marginals[seq_nb][pretty_name] = marginals[seq_nb].squeeze().tolist()
                evals_additive[seq_nb][pretty_name] = additive[seq_nb].squeeze().tolist()

evals_additive = pd.DataFrame.from_dict(evals_additive, orient="index").stack().to_frame()

# to break out the lists into columns
evals_additive = pd.DataFrame(evals_additive[0].values.tolist(), index=evals_additive.index).T
evals_additive = evals_additive.unstack().reset_index()
slices_length = slices[-1] - slices[-2]
evals_additive.columns = [f'{exp_type}', 'Model', 'Timestep', 'Additive error']
evals_additive['Timestep']*=slices_length
fig, ax = plt.subplots(1,1)
sns.lineplot(ax=ax, data=evals_additive, x='Timestep', y='Additive error', hue='Model', errorbar=None)
handles, labels = ax.get_legend_handles_labels()
sns.lineplot(ax=ax, data=evals_additive, x='Timestep', y='Additive error', hue='Model', style=f'{exp_type}', alpha=0.3)
ax.legend(handles, labels)
plt.savefig(os.path.join(eval_dir,'additive_error'))
plt.savefig(os.path.join(eval_dir,'additive_error.pdf'),format='pdf')

plt.close()


evals_marginal = pd.DataFrame.from_dict(evals_marginals, orient="index").stack().to_frame()

# to break out the lists into columns
evals_marginal = pd.DataFrame(evals_marginal[0].values.tolist(), index=evals_marginal.index).T
evals_marginal = evals_marginal.unstack().reset_index()
evals_marginal.columns = [f'{exp_type}', 'Model', 'Timestep', 'Marginal error']

sns.lineplot(data=evals_marginal, x='Timestep', y='Marginal error',  hue='Model', alpha=1)

plt.savefig(os.path.join(eval_dir,'Marginal_error'))
plt.savefig(os.path.join(eval_dir,'Marginal_error.pdf'), format='pdf')

plt.close()


training_curve_file = ''