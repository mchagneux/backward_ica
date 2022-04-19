# Experiments on backward ICA

## Local installation 

1. Create an environment:
```shell 
python3 -m venv <path-to-your-new-env>
source <path-to-your-new-env>/bin/activate
pip install --upgrade pip
``` 
2. Install JAX
- If you have a GPU with CUDA >= 11.1 supported and corresponding driver
```shell
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

- Otherwise

```shell
pip install --upgrade "jax[cpu]"
```

*Refer to [JAX installation instructions](https://github.com/google/jax#installation) for more informations.*

3. Install remaining dependencies: 

```shell 
pip install -r requirements.txt
```






