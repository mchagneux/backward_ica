import jax, jax.numpy as jnp
from src.utils.misc import Serializer
from src.variational.inference_nets import build_model
import os
import datetime
import dataclasses
from src.utils.video_datasets import load_dataset


date = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')


@dataclasses.dataclass
class Config: 
  vae_path: str = 'experiments/gaussian_vae_training/2023_04_18__10_29_12'
  output_dir: str = os.path.join('datasets', date)


config = Config()

os.makedirs(config.output_dir)
Serializer(config.output_dir).save_config(config)

serializer_vae = Serializer(config.vae_path)
config_vae = serializer_vae.load_config()
vae_weights = serializer_vae.load_params()
vae_model = lambda batch: build_model(config=config_vae) \
                            .apply(vae_weights, jax.random.PRNGKey(0), batch) \
                            .mean_z


dataset_path = config_vae['dataset_path']

ds = load_dataset(dataset_path,
                  split='train',
                  batch_size=128,
                  seed=0,
                  repeat=False)

results = []
frame_nbs = []
images = []
for batch in ds:
  images.append(batch.image)
  latents = vae_model(batch.image)
  results.append(latents)
  frame_nbs.append(batch.frame_nb)
  
latents = jnp.concatenate(results)
frame_nbs = jnp.concatenate(frame_nbs)
images = jnp.concatenate(images)

latents = jnp.concatenate([x for _, x in sorted(zip(frame_nbs, latents), 
                                          key=lambda pair: pair[0])])

images = jnp.concatenate([x for _, x in sorted(zip(frame_nbs, images), 
                                          key=lambda pair: pair[0])])



jnp.save(os.path.join(config.output_dir, 'X.npy'), latents)
jnp.save(os.path.join(config.output_dir, 'Y.npy'), images)

