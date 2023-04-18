import haiku as hk 
from jax import numpy as jnp, nn 
from src.utils.misc import *
from src.stats.distributions import * 
from src.stats.kernels import * 
from jax.flatten_util import ravel_pytree
from typing import NamedTuple, Sequence, Tuple
import dataclasses


## sequential stuff 
def deep_gru(obs, prev_state, layers):

    gru = hk.DeepRNN([hk.GRU(hidden_size) for hidden_size in layers])

    return gru(obs, prev_state)

def gaussian_proj(state, d):

    net = hk.Linear(2*d,
        w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
        b_init=hk.initializers.RandomNormal())


    eta1, out2 = jnp.split(net(state.out),2)
    eta2 = -jnp.diag(nn.softplus(out2))# - jnp.eye(d)
    
    # eta1, out2 = out[:d], out[d:]
    # # eta2_chol = chol_from_vec(out2, d)

    # # eta2 = - (eta2_chol @ eta2_chol.T + jnp.eye(1))
    # eta2 = -jnp.diag(nn.relu(out2))

    return Gaussian.Params(
                    eta1=eta1, 
                    eta2=eta2)

def backwd_net(aux, obs, layers, state_dim):
    d = state_dim

    
    net = hk.nets.MLP((*layers, 2*d),
        w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
        b_init=hk.initializers.RandomNormal(),
        activate_final=False)
    
    eta1, out2 = jnp.split(net(obs),2)
    eta2 = -jnp.diag(nn.softplus(out2))# - jnp.eye(d)

    # eta_2_chol = jnp.diagonal(nn.softplus(out2))
    # eta2 = -(eta_2_chol @ eta_2_chol.T + jnp.eye(d))
    # eta2 = -jnp.diag(nn.softplus(out2))

    return eta1, eta2

def johnson_anisotropic(obs, layers, state_dim):


    d = state_dim 
    out_dim = d + (d * (d+1)) // 2
    rec_net = hk.nets.MLP((*layers, out_dim),
                w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
                b_init=hk.initializers.RandomNormal(),
                activation=nn.tanh,
                activate_final=False)


    out = rec_net(obs)
    eta1 = out[:d]
    eta2 = mat_from_chol_vec(out[d:],d)

    return eta1, eta2

def johnson(obs, layers, state_dim):


    d = state_dim 
    rec_net = hk.nets.MLP((*layers, 2*d),
                w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
                b_init=hk.initializers.RandomNormal(),
                activation=nn.tanh,
                activate_final=False)


    eta1, out2 = jnp.split(rec_net(obs), 2)
    eta2 = -jnp.diag(nn.softplus(out2))
    return eta1, eta2

## Vanilla-VAE 

@dataclasses.dataclass
class Encoder(hk.Module):
  """Encoder model."""

  latent_size: int = 10
  hidden_size: int = 512

  def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Encodes an image as an isotropic Guassian latent code."""
    x = hk.Flatten()(x)
    x = hk.Linear(self.hidden_size)(x)
    x = jax.nn.relu(x)

    mean = hk.Linear(self.latent_size)(x)
    log_stddev = hk.Linear(self.latent_size)(x)
    stddev = jnp.exp(log_stddev)

    return mean, stddev
  

@dataclasses.dataclass
class Decoder(hk.Module):
  """Decoder model."""

  output_shape: Sequence[int]
  hidden_size: int = 512

  def __call__(self, z: jax.Array) -> jax.Array:
    """Decodes a latent code into Bernoulli log-odds over an output image."""
    z = hk.Linear(self.hidden_size)(z)
    z = jax.nn.relu(z)

    out = hk.Linear(np.prod(self.output_shape))(z)
    out = jnp.reshape(out, (-1, *self.output_shape))

    return out
  
  def build_model(config):

    def model(latent):
      decoder = Decoder(config['decoder_hidden_size'])
      return decoder(latent)
    
    return hk.without_apply_rng(hk.transform(model))




class BernoulliVAEOutput(NamedTuple):
  image: jax.Array
  mean: jax.Array
  variance: jax.Array
  logits: jax.Array

@dataclasses.dataclass
class BernoulliVariationalAutoEncoder(hk.Module):
  """Main VAE model class."""

  encoder: Encoder
  decoder: Decoder

  def __call__(self, x: jax.Array) -> BernoulliVAEOutput:
    """Forward pass of the variational autoencoder."""
    x = x.astype(jnp.float32)
    mean, stddev = self.encoder(x)
    z = mean + stddev * jax.random.normal(hk.next_rng_key(), mean.shape)
    logits = self.decoder(z)

    p = jax.nn.sigmoid(logits)
    image = jax.random.bernoulli(hk.next_rng_key(), p)

    return BernoulliVAEOutput(image, mean, jnp.square(stddev), logits)


class GaussianVAEOutput(NamedTuple):
  image: jax.Array
  mean_z: jax.Array
  variance_z: jax.Array
  mean_x: jax.Array
  variance_x: jax.Array

@dataclasses.dataclass
class GaussianVariationalAutoEncoder(hk.Module):
  """Main VAE model class."""

  encoder: Encoder
  decoder: Decoder

  def __call__(self, x: jax.Array) -> GaussianVAEOutput:
    """Forward pass of the variational autoencoder."""
    x = x.astype(jnp.float32)
    mean_z, stddev_z = self.encoder(x)
    z = mean_z + stddev_z * jax.random.normal(hk.next_rng_key(), mean_z.shape)
    mean_x = jax.nn.sigmoid(self.decoder(z))

    return GaussianVAEOutput(image=mean_x, 
                             mean_z=mean_z, 
                             variance_z=jnp.square(stddev_z), 
                             mean_x=mean_x,
                             variance_x=jnp.ones_like(mean_x) * 0.1)
  
def build_model(latent_size=None,
                encoder_hidden_size=None, 
                decoder_hidden_size=None,
                bernoulli=False,
                config=None):
  if config is not None: 
    latent_size = config['latent_size']
    encoder_hidden_size = config['encoder_hidden_size']
    decoder_hidden_size = config['decoder_hidden_size']
    bernoulli = config['vae_type'] == 'bernoulli'

  if bernoulli: 
    @hk.transform
    def model(x):
      vae = BernoulliVariationalAutoEncoder(
          encoder=Encoder(latent_size=10, 
                          hidden_size=encoder_hidden_size),
          decoder=Decoder(output_shape=x.shape[1:], 
                                   hidden_size=decoder_hidden_size),
      )
      return vae(x)
    
  else:
    @hk.transform
    def model(x):
      vae = GaussianVariationalAutoEncoder(
          encoder=Encoder(latent_size=latent_size, 
                          hidden_size=encoder_hidden_size),
          decoder=Decoder(output_shape=x.shape[1:], 
                                  hidden_size=decoder_hidden_size),
      )
      return vae(x)
  
  return model




#     A_back_dim = (d * (d+1)) // 2
#     a_back_dim = d
#     Sigma_back_dim = (d * (d + 1)) // 2
    
#     out_dim = A_back_dim + a_back_dim + Sigma_back_dim
    
#     net = hk.nets.MLP(output_sizes=(8,8,out_dim),
#                 activation=nn.tanh,
#                 activate_final=False, 
#                 w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
#                 b_init=hk.initializers.RandomNormal(),)

#     out = net(jnp.concatenate(state.hidden))

#     A_back_chol = chol_from_vec(out[:A_back_dim], d)
#     a_back = out[A_back_dim:A_back_dim+a_back_dim]
#     Sigma_back_vec = out[A_back_dim+a_back_dim:]

#     return Kernel.Params(map=Maps.LinearMapParams(w=A_back_chol @ A_back_chol.T + jnp.eye(d), b=a_back), 
#                         noise=Gaussian.NoiseParams.from_vec(Sigma_back_vec, d, chol_add=jnp.eye))