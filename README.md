# Paper

## Installation 
### Create en environment and activate it
```shell
python3 -m venv <location-of-env>
source <location-of-env>/bin/activate
```
Or use Conda. 

### Install JAX 
```shell
pip install --upgrade pip
pip install --upgrade "jax[cpu]"
```
To enable GPU computations, use instead
```shell 
pip install --upgrade pip

# CUDA 12 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CUDA 11 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Install remaining dependencies 
```shell
cd <this-directory>
pip install -r requirements.txt
```


## Experiments 

Experiments for the paper are in the notebook of the root folder.

