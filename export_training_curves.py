#%%s
import tensorboard as tb 
import seaborn as sns 
import matplotlib.pyplot as plt
from packaging import version

#%%
experiment_id = '1RJotZo7QPOMdf64Vb7onA'
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
#%%
df = experiment.get_scalars()
#%%
df = df[df.tag == 'ELBO']
df['fit'] = int(df.run.str.split('fit_').str[1][0])
# df = df[df.fit == 0]
#%%
def get_method_from_run_name(run_name):
    if 'closed_form' in run_name:
        return 'Pathwise, analytical recursions'
    elif 'autodiff_on_backward' in run_name:
        return 'Pathwise, backward trajectory sampling'
    elif 'score' in run_name:
        return 'Score-based, approximate recursions'
    
def get_quantity_type(run_name):

    if ('monitor' in run_name.split('fit_')[1]) or ('autodiff_on_backward' in run_name):
        return 'Backward trajectory sampling'
    else:
        return 'Approximate recursions'
    
df['gradients'] = df['run'].apply(get_method_from_run_name)
df['elbo'] = df['run'].apply(get_quantity_type)
#%%
# df = df[df.step > 2000]
df = df.iloc[:,2:]
df.columns = ['Timestep', 'ELBO value', 'Fit', 'Gradients', 'ELBO']
# df = df[df.Method == 'Backward trajectory saquantitympling']
#%%Fits
fig, ax = plt.subplots(1,1, figsize=(15,8))
plt.autoscale(True)
# plt.tight_layout()
sns.lineplot(df, ax=ax, x='Timestep', y='ELBO value', style='ELBO', hue='Gradients')#, style='Monitor', hue='Method', estimatWO5hkgQVTMuJh28drMzEAgor=None) #, style='Analytical')
#%%
plt.savefig('training_curve_lgm_dim_10.pdf', format='pdf', dpi=500)
#%%
# sns.lineplot(df, x='step', y='value'quantity)
