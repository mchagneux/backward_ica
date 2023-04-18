#%%
import dataclasses
from typing import NamedTuple
from absl import app
from absl import flags
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import os
import datetime
import dill
from src.video_datasets import load_dataset, Batch
from src.variational.inference_nets import build_model
import json

from src.utils import Serializer
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_enable_x64', False)

date = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

train_dir = os.path.join('experiments','gaussian_vae_training', date)
os.makedirs(train_dir)
serializer = Serializer(train_dir)

@dataclasses.dataclass
class Config:
  vae_type:str = 'gaussian'
  latent_size: int = 10
  encoder_hidden_size: int = 512
  decoder_hidden_size: int = 512
  batch_size: int = 128
  learning_rate: float = 1e-3
  training_steps: int = 20000
  eval_every: int = 100
  seed: int = 0
  dataset_path: str = 'datasets/2023_04_17__17_36_33'

#%%


class TrainingState(NamedTuple):
  params: hk.Params
  opt_state: optax.OptState
  rng_key: jax.Array



def main(_):

  flags.FLAGS.alsologtostderr = True
  config = Config()

  serializer.save_config(config)


  model = build_model(latent_size=config.latent_size, 
                  encoder_hidden_size=config.encoder_hidden_size, 
                  decoder_hidden_size=config.decoder_hidden_size,
                  bernoulli=config.vae_type=='bernoulli')
  
  if config.vae_type == 'gaussian':
    @jax.jit
    def loss_fn(params, rng_key, batch: Batch) -> jax.Array:
      """ELBO loss: E_p[log(x)] - KL(d||q), where p ~ Be(0.5) and q ~ N(0,1)."""

      # Run the model on the inputs.
      _, mean_z, var_z, mean_x, var_x = model.apply(params, rng_key, batch.image)



      # Gaussian log-likelihood (assumes `image` is binarised).
      log_likelihood = jax.vmap(jax.scipy.stats.norm.logpdf)(batch.image.flatten(),
                                                              mean_x.flatten(),
                                                              var_x.flatten()).reshape(mean_x.shape)

      log_likelihood = jax.vmap(jnp.sum)(log_likelihood)

      # KL divergence between Gaussians N(mean, std) and N(0, 1).
      kl = 0.5 * jnp.sum(-jnp.log(var_z) - 1. + var_z + jnp.square(mean_z), axis=-1)

      # Loss is the negative evidence lower-bound.
      return -jnp.mean(log_likelihood - kl)
  else:
    @jax.jit
    def loss_fn(params, rng_key, batch: Batch) -> jax.Array:
      """ELBO loss: E_p[log(x)] - KL(d||q), where p ~ Be(0.5) and q ~ N(0,1)."""

      # Run the model on the inputs.
      _, mean, var, logits = model.apply(params, rng_key, batch.image)

      # Bernoulli log-likelihood (assumes `image` is binarised).
      log_likelihood = jnp.einsum(
          "b...->b", batch.image * logits - jnp.logaddexp(0., logits))

      # KL divergence between Gaussians N(mean, std) and N(0, 1).
      kl = 0.5 * jnp.sum(-jnp.log(var) - 1. + var + jnp.square(mean), axis=-1)

      # Loss is the negative evidence lower-bound.
      return -jnp.mean(log_likelihood - kl)


  optimizer = optax.adam(config.learning_rate)

  @jax.jit
  def update(state: TrainingState, batch: Batch) -> TrainingState:
    """Performs a single SGD step."""
    rng_key, next_rng_key = jax.random.split(state.rng_key)
    gradients = jax.grad(loss_fn)(state.params, rng_key, batch)
    updates, new_opt_state = optimizer.update(gradients, state.opt_state)
    new_params = optax.apply_updates(state.params, updates)
    return TrainingState(new_params, new_opt_state, next_rng_key)

  # Load datasets.
  train_dataset = load_dataset(config.dataset_path,
                               "train",
                               config.batch_size,
                               config.seed,
                               binarize=(config.vae_type=='bernoulli'))
  eval_datasets = {
      "train": load_dataset(config.dataset_path,
                            "train",
                            config.batch_size,
                            config.seed,
                            binarize=(config.vae_type=='bernoulli')),
      "valid": load_dataset(config.dataset_path,
                            "valid",
                            config.batch_size,
                            config.seed,
                            binarize=(config.vae_type=='bernoulli')),
  }

  # Initialise the training state.
  initial_rng_key = jax.random.PRNGKey(config.seed)
  initial_params = model.init(initial_rng_key, next(train_dataset).image)
  initial_opt_state = optimizer.init(initial_params)
  state = TrainingState(initial_params, initial_opt_state, initial_rng_key)

  log_writer = tf.summary.create_file_writer(os.path.join(train_dir,'log_files'))

  # Run training and evaluation.
  for step in range(config.training_steps):
    state = update(state, next(train_dataset))

    if step % config.eval_every == 0:
      for split, ds in eval_datasets.items():
        loss = loss_fn(state.params, state.rng_key, next(ds))
        elbo = -jax.device_get(loss).item()
        logging.info({
            "step": step,
            "split": split,
            "elbo":elbo,
        })
        with log_writer.as_default():
          tf.summary.scalar(f'elbo_{split}', elbo, step)

  serializer.save_params(state.params)


if __name__ == "__main__":
  app.run(main)
