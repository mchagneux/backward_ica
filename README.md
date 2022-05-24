# Codebase for the paper "Amortized backward variational inference for nonlinear state-space models"

## Installation 

1. Create an environment:
```shell 
python3 -m venv <path-to-your-new-env>
source <path-to-your-new-env>/bin/activate
pip install --upgrade pip
``` 
2. Install JAX
```shell
pip install --upgrade "jax[cpu]"
```

*In case of problems here, refer to [JAX installation instructions](https://github.com/google/jax#installation) for more informations.*

3. Install remaining dependencies: 

```shell 
pip install -r requirements.txt
```


## Reproducing the experiments of the paper

The code is divided into several files: 
- The core of the codebase is found in the subfolder [backward_ica](backward_ica).
- The experiment files are at the root folder. 

### 1. If you want to reproduce the experiments from section 5.1 of the paper:
1. Training: run [train_linear.py](train_linear.py). This will fit a linear Gaussian variational model with 5 different starting points, and select the best one based on the highest value of the ELBO. A folder will be created with path `experiments/p_linear/trainings/q_linear/<date>`. At this location, model parameters are saved and the training curves are plotted into image files.
2. Evaluation: Copy the full path of the training folder and paste it line 79 of the evaluation file [eval_linear.py](eval_linear.py)
```shell
train_path='<path-to-your-training-folder>'
```
then run the evaluation script. All results from the paper should be created into a folder with path `experiments/p_linear/eval/q_linear_<date-of-the-training>`.

### 2. If you want to reproduce the experiments from section 5.2

