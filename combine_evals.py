import pandas as pd 
import seaborn as sns 
import numpy as np 
import dill 
import os 
import matplotlib.pyplot as plt
base_eval_folder = 'experiments/p_chaotic_rnn/2022_10_20__11_53_44'

output_folder = os.path.join(base_eval_folder, 'eval_combined')
os.makedirs(output_folder, exist_ok=True)
evals_additive = dict()
evals_marginals = dict()
up_to = 200
with open(os.path.join(base_eval_folder, 'eval_johnson_forward','eval_external_campbell.dill'), 'rb') as f:
    evals = dill.load(f)
    evals_additive['Campbell'] = evals[2].squeeze()[:up_to]
    evals_marginals['Campbell'] = evals[1].squeeze()[:up_to]

methods = ['johnson_backward', 'johnson_forward', 'neural_backward_linear']
pretty_names = ['Johnson Backward', 'Johnson Forward', 'GRU Backward']

for method, pretty_name in zip(methods, pretty_names):
    with open(os.path.join(base_eval_folder, f'eval_{method}', f'eval_{method}.dill'), 'rb') as f: 
        evals = dill.load(f)
        evals_additive[pretty_name] = evals[2].squeeze()[:up_to]
        evals_marginals[pretty_name] = evals[1].squeeze()[:up_to]

evals_additive = pd.DataFrame(evals_additive).unstack().reset_index()
evals_additive.columns = ['Method', 'Timestep', 'Additive error']
evals_additive['Timestep'] = evals_additive['Timestep']*10 + 10
sns.lineplot(evals_additive, x='Timestep', y='Additive error', hue='Method')
plt.savefig(os.path.join(output_folder,'additive_error'))
plt.close()

evals_marginals = pd.DataFrame(evals_marginals).unstack().reset_index()
evals_marginals.columns = ['Method', 'Timestep', 'Marginal error']
evals_marginals['Timestep'] = evals_marginals['Timestep']*10 + 10
sns.lineplot(evals_marginals, x='Timestep', y='Marginal error', hue='Method')
plt.savefig(os.path.join(output_folder,'marginal_error'))

