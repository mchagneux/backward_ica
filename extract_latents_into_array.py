#%%
import jax, jax.numpy as jnp
from src.utils.misc import Serializer
from src.variational.inference_nets import build_model, Decoder
import os
import datetime
import dataclasses
from src.utils.video_datasets import load_dataset
from src.stats.hmm import LinearGaussianHMM
key = jax.random.PRNGKey(0)

date = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')


@dataclasses.dataclass
class Config: 
  vae_path: str = 'experiments/gaussian_vae_training/2023_04_18__10_29_12'
  output_dir: str = os.path.join('data', date)


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
                  batch_size=128,
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

  return latents, images


latent_sequence, reconstructed_images = extract_latents(vae_encoder, ds)

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



vae_encoder = Decoder(latent_size=config_vae['latent_size'], 
                      hidden_size=config_vae['decoder_hidden_size'])




# X_smoothed = p.smooth_seq(Y.squeeze(), 
#                           fitted_params)#%%




