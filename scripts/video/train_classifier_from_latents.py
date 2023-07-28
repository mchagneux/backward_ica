# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A minimal MNIST classifier example."""

from typing import Iterator, NamedTuple

from absl import app
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
import dill 
import os 
from src.variational.inference_nets import build_model
from src.utils.video_datasets import load_dataset
import dataclasses
import datetime
from src.utils.misc import Serializer
from src.utils.misc import classify_fn, NUM_CLASSES

date = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

train_dir = os.path.join('experiments','latent_classif_training', date)
os.makedirs(train_dir)

@dataclasses.dataclass
class Config:
  batch_size: int = 128
  learning_rate: float = 1e-3
  training_steps: int = 5000
  eval_every: int = 100
  seed: int = 0
  pretrained_vae_path: str = 'experiments/gaussian_vae_training/2023_04_17__17_39_06'
  dataset_path: str = 'datasets/2023_04_17__17_36_33'

config = Config()
serializer_classif = Serializer(train_dir)
serializer_vae = Serializer(config.pretrained_vae_path)

serializer_classif.save_config(config)
config_vae = serializer_vae.load_config()
pretrained_weights =  serializer_vae.load_params()

frozen_encoder = lambda image_batch: build_model(config=config_vae).apply(pretrained_weights, 
                                                                        jax.random.PRNGKey(0),
                                                                        image_batch).mean_z



class Batch(NamedTuple):
  image: np.ndarray  # [B, H, W, 1]
  label: np.ndarray  # [B]


class TrainingState(NamedTuple):
  params: hk.Params
  avg_params: hk.Params
  opt_state: optax.OptState




def main(_):
  # First, make the network and optimiser.
  network = hk.without_apply_rng(hk.transform(classify_fn))
  optimiser = optax.adam(1e-3)

  def loss(params: hk.Params, batch: Batch) -> jax.Array:
    """Cross-entropy classification loss, regularised by L2 weight decay."""
    batch_size, *_ = batch.image.shape
    logits = network.apply(params, frozen_encoder(batch.image))
    labels = jax.nn.one_hot(batch.label, NUM_CLASSES)

    l2_regulariser = 0.5 * sum(
        jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
    log_likelihood = jnp.sum(labels * jax.nn.log_softmax(logits))

    return -log_likelihood / batch_size + 1e-4 * l2_regulariser

  @jax.jit
  def evaluate(params: hk.Params, batch: Batch) -> jax.Array:
    """Evaluation metric (classification accuracy)."""
    logits = network.apply(params, frozen_encoder(batch.image))
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == batch.label)

  @jax.jit
  def update(state: TrainingState, batch: Batch) -> TrainingState:
    """Learning rule (stochastic gradient descent)."""
    grads = jax.grad(loss)(state.params, batch)
    updates, opt_state = optimiser.update(grads, state.opt_state)
    params = optax.apply_updates(state.params, updates)
    # Compute avg_params, the exponential moving average of the "live" params.
    # We use this only for evaluation (cf. https://doi.org/10.1137/0330046).
    avg_params = optax.incremental_update(
        params, state.avg_params, step_size=0.001)
    return TrainingState(params, avg_params, opt_state)

  # Make datasets.
  train_dataset = load_dataset(path=config.dataset_path, 
                               split="train", 
                               batch_size=config.batch_size, 
                               seed=config.seed)
  eval_datasets = {
      split: load_dataset(path=config.dataset_path, 
                          split=split, 
                          batch_size=config.batch_size, 
                          seed=config.seed)
      for split in ("train", "valid")
  }

  # Initialise network and optimiser; note we draw an input to get shapes.
  initial_params = network.init(
      jax.random.PRNGKey(seed=0), frozen_encoder(next(train_dataset).image))
  initial_opt_state = optimiser.init(initial_params)
  state = TrainingState(initial_params, initial_params, initial_opt_state)

  # Training & evaluation loop.
  for step in range(config.training_steps):
    if step % 100 == 0:
      # Periodically evaluate classification accuracy on train & test sets.
      # Note that each evaluation is only on a (large) batch.
      for split, dataset in eval_datasets.items():
        accuracy = np.array(evaluate(state.avg_params, next(dataset))).item()
        print({"step": step, "split": split, "accuracy": f"{accuracy:.3f}"})

    # Do SGD on a batch of training examples.
    state = update(state, next(train_dataset))

  with open(os.path.join(train_dir, 'trained_weights'), 'wb') as f: 
    dill.dump(state.params, f)

if __name__ == "__main__":
  app.run(main)