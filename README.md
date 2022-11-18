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
