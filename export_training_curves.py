#%%s
import tensorboard as tb 
import seaborn as sns 
import matplotlib.pyplot as plt
experiment_id = 'AMNPtyvpT42UnUwqwC85wQ'
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()
df = df[df.tag == 'ELBO']
df['fit'] = df.run.str.split('fit_').str[1]

def get_method_from_run_name(run_name):
    if 'autodiff_on_backward' in run_name:
        return 'Backward trajectory sampling'
    elif 'score' in run_name:
        return 'Recursive'
df['method'] = df['run'].apply(get_method_from_run_name)
# df = df[df.method == 'Backward trajectory sampling']
df = df[df.step > 100]
df = df.iloc[:,2:]
df.columns = ['Epoch', 'ELBO', 'Fit', 'Method']
#%%
sns.lineplot(df, x='Epoch', y='ELBO', hue='Method')
plt.savefig('training_curve_dim5_500obs.pdf', format='pdf')
#%%
# sns.lineplot(df, x='step', y='value')
