import pandas as pd 
import seaborn as sns 
import numpy as np 
import dill 
import os 
import matplotlib.pyplot as plt
from datetime import datetime
from backward_ica.utils import save_args
import argparse 


exp_type = 'Train mode'

exp_dirs = ['experiments/p_chaotic_rnn/2022_10_27__15_14_30',
            'experiments/p_chaotic_rnn/2022_10_27__15_18_19']

exp_names = ['All subsequences',
            'Whole subsequence only']
            
date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

eval_dir = os.path.join('experiments', 'combine_evals', date)
os.makedirs(eval_dir, exist_ok=True)

evals_additive = dict()
evals_marginals = dict()

method_names = ['neural_backward_linear', 
                'johnson_backward', 
                'external_campbell']


args = argparse.Namespace()
args.exp_dirs = exp_dirs 
args.method_names = method_names 
args.eval_dir = eval_dir

save_args(args, 'args', eval_dir)
up_to = 100
#,'neural_backward_linear']

for exp_name, exp_dir in zip(exp_names, exp_dirs):

    evals_marginals[exp_name] = dict()
    evals_additive[exp_name] = dict()

    for method_name in method_names: 
        method_eval_dir = os.path.join(exp_dir, method_name, 'eval')
        if method_name == 'johnson_backward':
            pretty_name = 'Conjugate Backward'
        elif method_name == 'johnson_forward':
            pretty_name = 'Conjugate Forward'
        elif method_name == 'neural_backward_linear':
            pretty_name = 'GRU Backward'
        elif method_name == 'external_campbell':
            pretty_name = 'Campbell'

        with open(os.path.join(method_eval_dir, f'eval.dill'), 'rb') as f: 
            evals = dill.load(f)
            evals_marginals[exp_name][pretty_name] = evals[0].squeeze().tolist()[:up_to]
            evals_additive[exp_name][pretty_name] = evals[1].squeeze().tolist()[:up_to]

evals_additive = pd.DataFrame.from_dict(evals_additive, orient="index").stack().to_frame()
# to break out the lists into columns
evals_additive = pd.DataFrame(evals_additive[0].values.tolist(), index=evals_additive.index).T
evals_additive = evals_additive.unstack().reset_index()
evals_additive.columns = [f'{exp_type}', 'Model', 'Timestep', 'Additive error']

fig, ax = plt.subplots(1,1)
sns.lineplot(ax=ax, data=evals_additive, x='Timestep', y='Additive error', hue='Model', style=f'{exp_type}', alpha=1)
handles, labels = ax.get_legend_handles_labels()
# sns.lineplot(ax=ax, data=evals_additive, x='Timestep', y='Additive error', hue='Model')
ax.legend(handles, labels)
plt.savefig(os.path.join(eval_dir,'additive_error'))
plt.savefig(os.path.join(eval_dir,'additive_error.pdf'),format='pdf')

plt.close()


evals_marginal = pd.DataFrame.from_dict(evals_marginals, orient="index").stack().to_frame()
# to break out the lists into columns
evals_marginal = pd.DataFrame(evals_marginal[0].values.tolist(), index=evals_marginal.index).T
evals_marginal = evals_marginal.unstack().reset_index()
evals_marginal.columns = [f'{exp_type}', 'Model', 'Timestep', 'Marginal error']

sns.lineplot(data=evals_marginal, x='Timestep', y='Marginal error',  hue='Model', style=f'{exp_type}', alpha=1)

plt.savefig(os.path.join(eval_dir,'Marginal_error'))
plt.savefig(os.path.join(eval_dir,'Marginal_error.pdf'), format='pdf')

plt.close()
