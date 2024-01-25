import jax, jax.numpy as jnp
from jax_tqdm import scan_tqdm


class ULA:

    def __init__(self, 
                 p,
                 num_steps=10_000,
                 h=1e-3, 
                 num_particles=50):

        self.p = p
        self.num_steps = num_steps
        self.h = h
        self.num_particles = num_particles

    def smooth(self, key, y, params):


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
        keys = jax.random.split(key, self.num_steps)

        x_init = jax.jit(jax.vmap(self.p.sample_prior, in_axes=(0,None,None)), static_argnums=2)(
                                                jax.random.split(key_init, self.num_particles), 
                                                params, 
                                                len(y)) # sample N trajectories samples from the hidden chain
        

        @scan_tqdm(len(keys))
        def _step(x, input):
            _, key = input
            grad_log_l = jax.vmap(jax.grad(joint_logl))(x) # vmapping over particles the fn that computes the gradient
            x += self.h*grad_log_l + jnp.sqrt(2*self.h)*jax.random.normal(key, shape=(self.num_particles, 
                                                                        len(y), 
                                                                        self.p.state_dim))
            return x, None


        
        steps = jnp.arange(len(keys)) # needed to get a progress bar in scan

        x_end = jax.lax.scan(
                            _step, 
                            init=x_init, 
                            xs=(steps, keys))[0]


        x_pred = jnp.mean(x_end, 
                            axis=0)

        return x_pred 
    
    def filt(self, key, y, params):
        formatted_params = self.p.format_params(params)

        num_steps_per_timstep = self.num_steps // y.shape[0]

        key, key_init = jax.random.split(key, 2)

        x_init = jax.jit(jax.vmap(self.p.prior_dist.sample, in_axes=(0,None)))(jax.random.split(key_init, self.num_particles), 
                                            formatted_params.prior)
        
        @scan_tqdm(y.shape[0])
        def _temporal_step(x_tm1, t_key_and_y):
            t, key, y_t = t_key_and_y


            def _init_step(x_0, key): # basic langevin to target \phi_0 \propto p(y_0|x_0)p(x_0)
                def _ula_step(x_t, key):
                    def _gradient_term(x_t_i):
                        return self.p.emission_kernel.logpdf(y_t, x_t_i, formatted_params.emission) \
                                + self.p.prior_dist.logpdf(x_t_i, formatted_params.prior)
                    
                    grad = jax.vmap(jax.grad(_gradient_term))(x_t)
                    x_t += self.h*grad + jnp.sqrt(2*self.h)*jax.random.normal(key, shape=(self.num_particles, 
                                                                                self.p.state_dim))
                    return x_t, None
                
                return jax.lax.scan(_ula_step, 
                                init=x_0, 
                                xs=(jax.random.split(key, num_steps_per_timstep)))[0]
            

            def _advance_step(x_tm1, key): 
                '''Langevin to target \phi_t \propto \int p(y_t|x_t)p(x_t|d x_{t-1}) \phi_{t-1}(d x_{t-1})
                where \phi_{t-1} is replaced with 1 /  N \sum_{i=1}^N choo
                '''


                key, key_init = jax.random.split(key, 2)
                x_t_init = jax.vmap(self.p.transition_kernel.sample, in_axes=(0,0,None))(jax.random.split(key_init, 
                                                                                                    self.num_particles), 
                                                                                                x_tm1, 
                                                                                                formatted_params.transition)
                def _ula_step(x_t, key):
                    def _gradient_term(x_t_i):
                        def _sum_component(x_t_i, x_tm1_j):
                            return self.p.transition_kernel.logpdf(x_t_i, x_tm1_j, formatted_params.transition)
                    
                        return self.p.emission_kernel.logpdf(y_t, x_t_i, formatted_params.emission) \
                            + jax.scipy.special.logsumexp(jax.vmap(_sum_component, in_axes=(None, 0))(x_t_i, x_tm1))
                    
                    grad = jax.vmap(jax.grad(_gradient_term))(x_t)
                    x_t += self.h*grad + jnp.sqrt(2*self.h)*jax.random.normal(key, shape=(self.num_particles, 
                                                                                self.p.state_dim))
                    return x_t, None

                return jax.lax.scan(_ula_step, 
                                    init=x_t_init, 
                                    xs=(jax.random.split(key, num_steps_per_timstep)))[0]
            
            x_t = jax.lax.cond(t > 0, _advance_step, _init_step, x_tm1, key)
            return x_t, jnp.mean(x_t, axis=0)
    
        x_pred = jax.lax.scan(_temporal_step, 
                                init=x_init,
                                xs=(jnp.arange(0, y.shape[0]), 
                                    jax.random.split(key, y.shape[0]), 
                                    y))[1]


        return x_pred
        
    def learn_params(self, key, y, params_init):


        key, key_init = jax.random.split(key, 2)

        x_init = jax.jit(jax.vmap(self.p.sample_prior, in_axes=(0,None,None)), static_argnums=2)(
                                                jax.random.split(key_init, self.num_particles), 
                                                params_init, 
                                                len(y)) # sample N trajectories samples from the hidden chain
        

        def joint_logl(x, params):
            formatted_params = self.p.format_params(params)
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
    


        keys = jax.random.split(key, self.num_steps)


        @scan_tqdm((len(keys)))
        def _step(carry, input):

            x, params = carry
            _, key = input

            
            # advance langevin one step
            grad_log_l_wrt_x = jax.vmap(jax.grad(joint_logl, argnums=0), in_axes=(0,None))(x, params) # vmapping over particles the fn that computes the gradient
            x += self.h*grad_log_l_wrt_x + jnp.sqrt(2*self.h)*jax.random.normal(key, shape=(self.num_particles, 
                                                                        len(y), 
                                                                        self.p.state_dim))
            
            grad_log_l_wrt_params = jax.vmap(jax.grad(joint_logl, argnums=1), 
                                             in_axes=(0,None))(x, params)
            
            params = jax.tree_map(lambda x,y: x + self.h*jnp.mean(y, axis=0), 
                                  params, grad_log_l_wrt_params)


            return (x, params), None
        

        
        steps = jnp.arange(len(keys)) # needed to get a progress bar in scan

        fitted_params = jax.lax.scan(
                            _step, 
                            init=(x_init, params_init),
                            xs=(steps, keys))[0]



        return fitted_params