from src.misc import QuadForm
from src.elbo import linear_gaussian_elbo
from src.hmm import LinearGaussianHMM
from src.kalman import NumpyKalman
from src.kalman import filter
from jax.random import PRNGKey
import jax.numpy as jnp
import jax

quad_forms = QuadForm(A=jnp.array([jnp.eye(2)]*5), b=jnp.array([jnp.ones(2)]*5), Omega=jnp.array([jnp.eye(2)]*5))


# def function_applied_to_quad_form(quad_form, nb_to_add):
#     A = quad_form.A + nb_to_add
#     b = quad_form.b + nb_to_add
#     Omega = quad_form.Omega + nb_to_add
#     return QuadForm(A=A, b=b, Omega=Omega)


# function_applied_to_quad_forms = jax.vmap(function_applied_to_quad_form, in_axes=(0,None))

# transformed_quad_forms = function_applied_to_quad_forms(quad_forms, 1)

A = quad_forms.A.at[3].set(2*jnp.eye(2))

quad_forms = QuadForm(A=A, b=quad_forms.b, Omega=quad_forms.Omega)
print(quad_forms.A[3])

# key = PRNGKey(0)
# key, model = LinearGaussianHMM.get_random_model(key=key, state_dim=2, obs_dim=2)
# key, states, observations = LinearGaussianHMM.sample_joint_sequence(key, model, 10)
# _, _, filtered_means, filtered_covs, likelihood = filter(model=model, observations=observations)
# filtered_means_numpy, filtered_covs_numpy, likelihood_numpy = NumpyKalman(model).filter(observations)

# elbo = linear_gaussian_elbo(model, model, observations)

# print(likelihood - likelihood_numpy)
# print(jnp.abs(elbo - likelihood))