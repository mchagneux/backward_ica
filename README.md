# Backward ICA codebase


## Initial remarks
This is a first, quickly written implementation of research work on "backward ICA" algorithms under a variety of models with own wrappers around SciPy and NumPy objects + Kalman recursions written by hand. The natural direction for this kind of work will be:

- First, turn relevant objects into JAX or Pytorch objects to perform automatic differentiation (JAX conversion should be faster)
- If good results are found, rewrite everything using a pre-built probabilistic programming framework like Pyro (or the new NumPyro with JAX support) which should be faster.
- Vectorize operations as much as possible


## Installation 

```shell 
conda env create -f environment.yml
``` 

## Documentation 








