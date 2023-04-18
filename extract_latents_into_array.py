#%%
import jax, jax.numpy as jnp
from src.utils.misc import Serializer
from src.variational.inference_nets import build_model, Decoder, build_decoder
import os
import datetime
import dataclasses
from src.utils.video_datasets import load_dataset
from src.stats.hmm import LinearGaussianHMM
key = jax.random.PRNGKey(0)
# jax.config.update('jax_platform_name', 'cpu')

date = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')


@dataclasses.dataclass
class Config: 
  vae_path: str = 'experiments/gaussian_vae_training/2023_04_18__10_29_12'
  output_dir: str = os.path.join('experiments', 'latent_smoothing', date)


config = Config()

os.makedirs(config.output_dir)
Serializer(config.output_dir).save_config(config)

serializer_vae = Serializer(config.vae_path)
config_vae = serializer_vae.load_config()
vae_weights = serializer_vae.load_params()
vae_encoder = lambda batch: build_model(config=config_vae) \
                            .apply(vae_weights, jax.random.PRNGKey(0), batch) \
                            .mean_z


dataset_path = config_vae['dataset_path']

ds = load_dataset(dataset_path,
                  split='train',
                  batch_size=1080,
                  seed=0,
                  repeat=False)


def extract_latents(model, dataset):

  results = []
  frame_nbs = []
  images = []
  for batch in dataset:
    images.append(batch.image)
    latents = model(batch.image)
    results.append(latents)
    frame_nbs.append(batch.frame_nb)
    
  results = jnp.concatenate(results)
  frame_nbs = jnp.concatenate(frame_nbs)
  images = jnp.concatenate(images)

  latents = jnp.array([x for _, x in sorted(zip(frame_nbs, results), 
                                            key=lambda pair: pair[0])])
  
  images = jnp.array([x for _, x in sorted(zip(frame_nbs, images), 
                                            key=lambda pair: pair[0])])

  return latents, images


latent_sequence, images = extract_latents(vae_encoder, ds)

p = LinearGaussianHMM(state_dim=config_vae['latent_size'], 
                      obs_dim=config_vae['latent_size'],
                      transition_matrix_conditionning='diagonal',
                      range_transition_map_params=(0.9,1),
                      transition_bias=True, 
                      emission_bias=False)

Y = latent_sequence[jnp.newaxis,:]

fitted_params, training_curve, _ = p.fit_kalman_rmle(
                                      key, 
                                      Y, 
                                      'adam', 
                                      1e-1, 
                                      1, 
                                      100)

smoothed_latents, _ = p.smooth_seq(latent_sequence, 
                                fitted_params)

import matplotlib.pyplot as plt
plt.plot(training_curve)
plt.savefig(os.path.join(config.output_dir, 'training_curve'))
#%%


vae_decoder = build_decoder(output_shape=images[0].shape[:-1], 
                            config=config_vae)

reconstructions_from_original_latents = vae_decoder.apply(vae_weights, key, latent_sequence)

reconstructions_from_smoothed_latents = vae_decoder.apply(vae_weights, key, smoothed_latents)

images_dir = os.path.join(config.output_dir, 'images')
os.makedirs(images_dir)
for image_nb, (image, reconstruction_from_original, reconstruction_from_smoothed) in enumerate(zip(images, 
                                                                                                   reconstructions_from_original_latents, 
                                                                                                   reconstructions_from_smoothed_latents)):
  fig, (ax0, ax1, ax2) = plt.subplots(1,3)
  ax0.imshow(image[...,0])
  ax0.set_title('Original image')

  ax1.imshow(reconstruction_from_original)
  ax1.set_title('Reconstruction from original latent')

  ax2.imshow(reconstruction_from_smoothed)
  ax2.set_title('Reconstruction from smoothed latent')
  plt.autoscale(True)
  plt.tight_layout()
  plt.savefig(os.path.join(images_dir, f'{image_nb}'))

#%%

# X_smoothed = p.smooth_seq(Y[0], 
#                           fitted_params)#%%




