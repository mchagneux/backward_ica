import numpy as np
from src.kalman import filter as kalman_filter
from src.kalman import Kalman as NumpyKalman
from src.misc import * 
import matplotlib.pyplot as plt 
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from src.linear_gaussian_hmm import sample_joint_sequence
from src.elbo import linear_gaussian_elbo

import jax
import jax.numpy as jnp
import jax.random
from jax.random import PRNGKey as generate_key
from jax.random import multivariate_normal, normal
from jax.config import config; config.update("jax_enable_x64", True)
import optax
from optax import GradientTransformation
from tqdm import tqdm
from time import time


## verbose functions 

def visualize_kalman_results(true_states, observations, filtered_state_means, filtered_state_covariances):
    
    for true_state, observation, filtered_state_mean, filtered_state_covariance in zip(true_states, observations, filtered_state_means, filtered_state_covariances): 
        plt.scatter(true_state[0], true_state[1], c='r')
        plt.scatter(observation[0], observation[1], c='b')
        plt.scatter(filtered_state_mean[0], filtered_state_mean[1], c='tab:purple')

        cov = filtered_state_covariance
        n_std = 2
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])

        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0,0),
            width=ell_radius_x * 2,
            height=ell_radius_y * 2, fill=False)

        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = filtered_state_mean[0]

        # calculating the stdandard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = filtered_state_mean[1]

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + plt.gca().transData)

        plt.gca().add_patch(ellipse)


    plt.savefig('kalman')

def test_kalman():

    model_params = get_model()
    key = generate_key(0)
    true_states, observations = sample_joint_sequence(key=key, sequence_length=20, model_params=model_params)

    filtered_state_means, filtered_state_covariances, loglikelihood = filter(observations, model_params)
    print(loglikelihood)
    filtered_state_means, filtered_state_covariances, loglikelihood = NumpyKalman(model_params).filter(observations)
    print(loglikelihood)
    visualize_kalman_results(true_states, observations, filtered_state_means, filtered_state_covariances)

def get_model():

    ### Sampling from linear Gaussian HMM 
    state_dim = 2
    obs_dim = 2

    a = 2*jnp.ones(state_dim)
    A = 0.5 * jnp.eye(state_dim)
    Q = jnp.array([[0.1,0],
                   [0,0.5]])

    B = jnp.eye(obs_dim)
    b = 0*jnp.ones(obs_dim)
    R = jnp.array([[0.2,0],
                   [0,0.3]])


    transition = Transition(matrix=A, offset=a, cov=Q)
    emission = Emission(matrix=B, offset=b, cov=R)
    prior = Prior(mean=a, cov=Q)

    return Model(transition=transition, 
                emission=emission, 
                prior=prior)

def get_random_model(key):

    state_dim, obs_dim = 2, 2
    key, *subkeys = jax.random.split(key, num=9)

    prior_mean = 0.1 * normal(subkeys[0],shape=(state_dim,)) ** 2
    prior_cov = 0.1 * jnp.diag(normal(subkeys[1], shape=(state_dim,))) ** 2

    transition_matrix = jnp.diag(normal(subkeys[2],shape=(state_dim,))) ** 2
    transition_offset = normal(subkeys[3],shape=(state_dim,))
    transition_cov = 0.1 * jnp.diag(normal(subkeys[4],shape=(state_dim,))) ** 2

    emission_matrix = normal(subkeys[5],shape=(obs_dim,obs_dim)) ** 2
    emission_offset = jnp.zeros((obs_dim,))
    emission_cov = 0.1 * jnp.diag(normal(subkeys[7],shape=(obs_dim,))) ** 2

    prior = Prior(mean=prior_mean, cov=prior_cov)

    transition = Transition(matrix=transition_matrix, offset=transition_offset, cov=transition_cov)
    emission = Emission(matrix=emission_matrix, offset=emission_offset, cov=emission_cov)
    return Model(prior, transition, emission)

# test_kalman()
model = get_model()
key = generate_key(0)
observation_sequences = [sample_joint_sequence(key=key, sequence_length=100, model_params=model)[1] for _ in range(10)]

fast_kf = jax.jit(kalman_filter)
print('Evidence:',fast_kf(observation_sequences[0] ,model)[2])
# print('Evidence, numpy:',NumpyKalman(model).filter(observations)[2])


# print('Evidence:',fast_kf(observations ,model)[2])
# print('Evidence, numpy:',NumpyKalman(model).filter(observations)[2])

# init_v_model = get_random_model(key)
# fast_elbo = linear_gaussian_elbo
# fast_grad = jax.jit(jax.grad(linear_gaussian_elbo, argnums=1))

# state_sequences = [sequence[0] for sequence in sequences]
# observation_sequences = [sequence[1] for sequence in sequences]
print('Init ELBO:', linear_gaussian_elbo(model, model, observation_sequences[0]))
# print('Grad ELBO:', fast_grad(model, model, observation_sequences[0]))

# for observations in observation_sequences:
#     print(fast_elbo(model, model, observations))

# print('ELBO fast:',fast_elbo(model, model, observations[2:]))
# optimizer = optax.adam(learning_rate = 1e-2)

# n_epochs = 3000

# def fit(params: optax.Params, optimizer: optax.GradientTransformation) -> optax.Params:
#     opt_state = optimizer.init(params)

#     @jax.jit
#     def step(params, opt_state, observations):
#         loss_value, grads = jax.value_and_grad(fast_elbo, argnums=1)(model, params, observations)
#         updates, opt_state = optimizer.update(grads, opt_state)
#         params = optax.apply_updates(params, updates)
#         return params, opt_state, loss_value

#     for _ in tqdm(range(n_epochs)):
#         params, opt_state, loss_value = step(params, opt_state, observations)

#     print('ELBO:', loss_value)

#     return params
  
# fitted_v_model = fit(init_v_model, optimizer)

# print(model)
# print(fitted_v_model)