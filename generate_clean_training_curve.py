#%%
import seaborn as sns
import pandas as pd
import os 
path = 'experiments/p_chaotic_rnn/2023_06_09__17_29_28'
sns.set_theme()
import matplotlib.pyplot as plt

training_curves = []
for file in os.listdir(path):
    if ('.csv' in file):
        training_curves.append(pd.read_csv(os.path.join(path, file))['Value'][50:])
        label = ''
        if 'score' in file:
            label = 'Score gradients'
        elif 'autodiff_on_backward' in file:
            label = 'Reparameterized backward gradients'

training_curves = pd.concat(training_curves, axis=1)

training_curves.columns = [
                        'Backward trajectory sampling',
                        'Recursive',
                           ]
sns.lineplot(training_curves)
plt.ylabel('ELBO')
plt.xlabel('Epoch')
plt.savefig(os.path.join(path, 'training_curve_dim_5_500obs.pdf'), format='pdf')
#%%
