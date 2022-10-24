import pandas as pd 
import seaborn as sns 
import numpy as np 
import dill 
import os 
import matplotlib.pyplot as plt

exp_dirs = ['experiments/p_chaotic_rnn/2022_10_24__17_15_25', 
            'experiments/p_chaotic_rnn/2022_10_24__17_25_51']

eval_dir = os.path.join('experiments', 'test')
os.makedirs(eval_dir, exist_ok=True)

evals_additive = dict()
evals_marginals = dict()
method_names = ['johnson_backward', 'johnson_forward', 'neural_backward_linear']

for exp_nb, exp_dir in enumerate(exp_dirs):

    evals_marginals[exp_nb] = dict()
    evals_additive[exp_nb] = dict()

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
            evals_marginals[exp_nb][pretty_name] = evals[0].squeeze().tolist()
            evals_additive[exp_nb][pretty_name] = evals[1].squeeze().tolist()

evals_additive = pd.DataFrame.from_dict(evals_additive, orient="index").stack().to_frame()
# to break out the lists into columns
evals_additive = pd.DataFrame(evals_additive[0].values.tolist(), index=evals_additive.index).T
evals_additive = evals_additive.unstack().reset_index()
evals_additive.columns = ['Exp', 'Model', 'Timestep', 'Additive error']

sns.lineplot(data=evals_additive, x='Timestep', y='Additive error', hue='Model')
plt.savefig(os.path.join(eval_dir,'additive_error'))
plt.close()

evals_marginal = pd.DataFrame.from_dict(evals_marginals, orient="index").stack().to_frame()
# to break out the lists into columns
evals_marginal = pd.DataFrame(evals_marginal[0].values.tolist(), index=evals_marginal.index).T
evals_marginal = evals_marginal.unstack().reset_index()
evals_marginal.columns = ['Exp', 'Model', 'Timestep', 'Marginal error']

sns.lineplot(data=evals_marginal, x='Timestep', y='Marginal error', hue='Model')
plt.savefig(os.path.join(eval_dir,'Marginal_error'))
plt.close()
