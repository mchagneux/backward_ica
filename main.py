from random import sample
import numpy as np
from utils.kalman import Kalman, filter

from utils.misc import ModelParams, TransitionParams, ObservationParams, PriorParams
import matplotlib.pyplot as plt 
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import jax.numpy as jnp
from jax import jit
from jax.random import PRNGKey as generate_key
from utils.linear_gaussian_hmm import sample_joint_sequence
from utils.elbo import compute as elbo_compute

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
    states, observations = sample_joint_sequence(key=key, sequence_length=20, model_params=model_params)

    filtered_state_means, filtered_state_covariances, loglikelihood = filter(observations, model_params)
    print(loglikelihood)
    filtered_state_means, filtered_state_covariances, loglikelihood = Kalman(model_params).filter(observations)
    print(loglikelihood)
    #visualize_kalman_results(true_states, observations, filtered_state_means, filtered_state_covariances)

def get_model():

    ### Sampling from linear Gaussian HMM 
    state_dim = 2
    obs_dim = 2

    a = 2*jnp.ones(state_dim)
    A = jnp.eye(state_dim)
    Q = jnp.array([[0.01,0],
                   [0,0.05]])

    B = jnp.eye(obs_dim)
    b = 0*jnp.ones(obs_dim)
    R = jnp.array([[0.01,0],
                   [0,0.01]])


    transition_params = TransitionParams(matrix=A, offset=a, cov=Q)
    observation_params = ObservationParams(matrix=B, offset=b, cov=R)
    prior_params = PriorParams(mean=a, cov=Q)

    return ModelParams(transition=transition_params, 
                                observation=observation_params, 
                                prior=prior_params)

model = get_model()
key = generate_key(0)
states, observations = sample_joint_sequence(key=key, sequence_length=20, model_params=model)


_, _, oracle_likelihood = Kalman(model).filter(observations)
_, _, jax_likelihood = filter(observations,model)

print('Oracle:',oracle_likelihood)
print('JAX oracle:',jax_likelihood)

v_model = model

elbo = elbo_compute(model, v_model, observations)

print('Elbo:', elbo)