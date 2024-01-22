import jax, jax.numpy as jnp
from jax_tqdm import scan_tqdm

class ULA:

    def __init__(self, 
                 p):

        self.p = p

    def fit(self, key, y, params, num_steps=10_000, h=1e-3, num_particles=50):


        formatted_params = self.p.format_params(params)
        def joint_logl(x):
            init_term = self.p.prior_dist.logpdf(x[0], formatted_params.prior)
            emission_terms = jax.vmap(self.p.emission_kernel.logpdf, 
                                        in_axes=(0,0,None))(
                                                        y, 
                                                        x,
                                                        formatted_params.emission)
            
            transition_terms = jax.vmap(self.p.transition_kernel.logpdf, 
                                        in_axes=(0,0,None))(x[1:], 
                                                            x[:-1], 
                                                            formatted_params.transition)
            return init_term + jnp.sum(emission_terms) + jnp.sum(transition_terms)

        key, key_init = jax.random.split(key, 2)
        keys = jax.random.split(key, num_steps)

        x_init = jax.jit(jax.vmap(self.p.sample_prior, in_axes=(0,None,None)), static_argnums=2)(
                                                jax.random.split(key_init, num_particles), 
                                                params, 
                                                len(y))
        

        scan_tqdm(len(keys))
        def _step(x, input):
            _, key = input
            grad_log_l = jax.vmap(jax.grad(joint_logl))(x)
            x += h*grad_log_l + jnp.sqrt(2*h)*jax.random.normal(key, shape=(num_particles, 
                                                                        len(y), 
                                                                        self.p.state_dim))
            return x, None


        
        steps = jnp.arange(len(keys))
        x_end = jax.lax.scan(
                            _step, 
                            init=x_init, 
                            xs=(steps, keys))[0]


        x_pred = jnp.mean(x_end, 
                            axis=0)

        return x_pred 
    


