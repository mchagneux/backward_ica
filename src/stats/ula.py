import jax, jax.numpy as jnp
from jax_tqdm import scan_tqdm


class ULA:

    def __init__(self, p, num_steps=10_000, h=1e-3, num_particles=50):
        """Initialize ULA (Unadjusted Langevin Algorithm) sampler.

        Args:
            p: Model object containing prior, transition and emission distributions
            num_steps: Number of Langevin steps to take. For smoothing, this is the total
                number of steps. For filtering, this is divided by the number of timesteps
                to get steps per timestep. Default: 10,000
            h: Step size for Langevin dynamics. Default: 0.001
            num_particles: Number of particles to use for approximating distributions.
                Default: 50

        The ULA sampler can be used in two modes:
        1. Smoothing (smooth method): Uses all observations to estimate full trajectories
        2. Filtering (filt method): Processes observations sequentially for online estimation

        For filtering, the actual number of Langevin steps per timestep will be
        num_steps // T, where T is the number of observations.
        """
        self.p = p
        self.num_steps = num_steps
        self.h = h
        self.num_particles = num_particles

    def smooth(self, key, y, params):

        formatted_params = self.p.format_params(params)

        def joint_logl(x):
            """Compute the joint log-likelihood of a trajectory.

            Args:
                x: Array of shape (T, state_dim) containing a trajectory of states.

            Returns:
                float: The joint log-likelihood, computed as:
                    log p(x_0) + sum_t log p(y_t|x_t) + sum_t log p(x_t|x_{t-1})
                where:
                - log p(x_0) is the log prior probability of the initial state
                - log p(y_t|x_t) are the emission log-likelihoods
                - log p(x_t|x_{t-1}) are the transition log-likelihoods
            """
            init_term = self.p.prior_dist.logpdf(x[0], formatted_params.prior)
            emission_terms = jax.vmap(
                self.p.emission_kernel.logpdf, in_axes=(0, 0, None)
            )(y, x, formatted_params.emission)

            transition_terms = jax.vmap(
                self.p.transition_kernel.logpdf, in_axes=(0, 0, None)
            )(x[1:], x[:-1], formatted_params.transition)
            return init_term + jnp.sum(emission_terms) + jnp.sum(transition_terms)

        key, key_init = jax.random.split(key, 2)
        keys = jax.random.split(key, self.num_steps)

        x_init = jax.jit(
            jax.vmap(self.p.sample_prior, in_axes=(0, None, None)), static_argnums=2
        )(
            jax.random.split(key_init, self.num_particles), params, len(y)
        )  # sample N trajectories samples from the hidden chain

        @scan_tqdm(len(keys))
        def _step(x, input):
            """Perform one step of Unadjusted Langevin Algorithm (ULA).

            Args:
                x: Array of shape (num_particles, T, state_dim) containing the current state
                   of all particles, where each particle is a full trajectory.
                input: Tuple of (step_index, random_key) where step_index is unused and
                       random_key is used for sampling noise.

            Returns:
                Tuple of:
                - Updated state x after one ULA step
                - None (required by scan but unused)
            """
            _, key = input
            grad_log_l = jax.vmap(jax.grad(joint_logl))(x)
            x += self.h * grad_log_l + jnp.sqrt(2 * self.h) * jax.random.normal(
                key, shape=(self.num_particles, len(y), self.p.state_dim)
            )
            return x, None

        # Create array of step indices for progress bar in scan
        steps = jnp.arange(len(keys))

        # Run ULA for specified number of steps using scan
        # Returns final state after all steps (x_end)
        x_end = jax.lax.scan(_step, init=x_init, xs=(steps, keys))[0]

        # Average across particles to get predicted trajectory
        x_pred = jnp.mean(x_end, axis=0)

        return x_pred

    def filt(self, key, y, params):
        """Perform online filtering using Langevin dynamics.

        This method implements a particle-based filtering algorithm that sequentially processes
        observations using Langevin dynamics. For each timestep, it updates particle positions
        to approximate the filtering distribution p(x_t|y_{0:t}).

        Args:
            key: JAX random key for random number generation
            y: Array of shape (T, obs_dim) containing the observations
            params: Parameters for the model (prior, transition, emission)

        Returns:
            x_pred: Array of shape (T, state_dim) containing the filtered state estimates,
                   computed by averaging particles at each timestep
        """
        # Format parameters into the expected structure
        formatted_params = self.p.format_params(params)

        # Calculate number of Langevin steps to take per timestep
        num_steps_per_timstep = self.num_steps // y.shape[0]

        # Split random key for initialization
        key, key_init = jax.random.split(key, 2)

        # Initialize particles by sampling from prior distribution
        x_init = jax.jit(jax.vmap(self.p.prior_dist.sample, in_axes=(0, None)))(
            jax.random.split(key_init, self.num_particles), formatted_params.prior
        )

        @scan_tqdm(y.shape[0])
        def _temporal_step(x_tm1, t_key_and_y):
            """Process one timestep of observations.

            Args:
                x_tm1: Particle positions from previous timestep
                t_key_and_y: Tuple of (timestep index, random key, observation)

            Returns:
                Tuple of (updated particles, mean particle position)
            """
            t, key, y_t = t_key_and_y

            def _init_step(x_0, key):
                """Initialize particles for first timestep using Langevin dynamics.

                Targets distribution proportional to p(y_0|x_0)p(x_0).
                """

                def _ula_step(x_t, key):
                    """Single step of Unadjusted Langevin Algorithm for initialization."""

                    def _gradient_term(x_t_i):
                        """Compute gradient of log posterior for a single particle."""
                        return self.p.emission_kernel.logpdf(
                            y_t, x_t_i, formatted_params.emission
                        ) + self.p.prior_dist.logpdf(x_t_i, formatted_params.prior)

                    grad = jax.vmap(jax.grad(_gradient_term))(x_t)
                    x_t += self.h * grad + jnp.sqrt(2 * self.h) * jax.random.normal(
                        key, shape=(self.num_particles, self.p.state_dim)
                    )
                    return x_t, None

                return jax.lax.scan(
                    _ula_step,
                    init=x_0,
                    xs=(jax.random.split(key, num_steps_per_timstep)),
                )[0]

            def _advance_step(x_tm1, key):
                """Advance particles using Langevin dynamics for timesteps after first.

                Targets the filtering distribution p(x_t|y_{0:t}) using particle approximation
                of previous filtering distribution.
                """
                # Split key and initialize particles using transition kernel
                key, key_init = jax.random.split(key, 2)
                x_t_init = jax.vmap(
                    self.p.transition_kernel.sample, in_axes=(0, 0, None)
                )(
                    jax.random.split(key_init, self.num_particles),
                    x_tm1,
                    formatted_params.transition,
                )

                def _ula_step(x_t, key):
                    """Single step of Unadjusted Langevin Algorithm for filtering."""

                    def _gradient_term(x_t_i):
                        """Compute gradient of log posterior for a single particle."""

                        def _sum_component(x_t_i, x_tm1_j):
                            """Compute transition term for a pair of current and previous particles."""
                            return self.p.transition_kernel.logpdf(
                                x_t_i, x_tm1_j, formatted_params.transition
                            )

                        return self.p.emission_kernel.logpdf(
                            y_t, x_t_i, formatted_params.emission
                        ) + jax.scipy.special.logsumexp(
                            jax.vmap(_sum_component, in_axes=(None, 0))(x_t_i, x_tm1)
                        )

                    grad = jax.vmap(jax.grad(_gradient_term))(x_t)
                    x_t += self.h * grad + jnp.sqrt(2 * self.h) * jax.random.normal(
                        key, shape=(self.num_particles, self.p.state_dim)
                    )
                    return x_t, None

                return jax.lax.scan(
                    _ula_step,
                    init=x_t_init,
                    xs=(jax.random.split(key, num_steps_per_timstep)),
                )[0]

            # Use initial step for t=0, advance step for t>0
            x_t = jax.lax.cond(t > 0, _advance_step, _init_step, x_tm1, key)
            return x_t, jnp.mean(x_t, axis=0)

        # Run filtering algorithm over all timesteps
        x_pred = jax.lax.scan(
            _temporal_step,
            init=x_init,
            xs=(jnp.arange(0, y.shape[0]), jax.random.split(key, y.shape[0]), y),
        )[1]

        return x_pred

    def learn_params(self, key, y, params_init):

        key, key_init = jax.random.split(key, 2)

        x_init = jax.jit(
            jax.vmap(self.p.sample_prior, in_axes=(0, None, None)), static_argnums=2
        )(
            jax.random.split(key_init, self.num_particles), params_init, len(y)
        )  # sample N trajectories samples from the hidden chain

        def joint_logl(x, params):
            formatted_params = self.p.format_params(params)
            init_term = self.p.prior_dist.logpdf(x[0], formatted_params.prior)
            emission_terms = jax.vmap(
                self.p.emission_kernel.logpdf, in_axes=(0, 0, None)
            )(y, x, formatted_params.emission)

            transition_terms = jax.vmap(
                self.p.transition_kernel.logpdf, in_axes=(0, 0, None)
            )(x[1:], x[:-1], formatted_params.transition)
            return init_term + jnp.sum(emission_terms) + jnp.sum(transition_terms)

        keys = jax.random.split(key, self.num_steps)

        @scan_tqdm((len(keys)))
        def _step(carry, input):

            x, params = carry
            _, key = input

            # advance langevin one step
            grad_log_l_wrt_x = jax.vmap(
                jax.grad(joint_logl, argnums=0), in_axes=(0, None)
            )(
                x, params
            )  # vmapping over particles the fn that computes the gradient
            x += self.h * grad_log_l_wrt_x + jnp.sqrt(2 * self.h) * jax.random.normal(
                key, shape=(self.num_particles, len(y), self.p.state_dim)
            )

            grad_log_l_wrt_params = jax.vmap(
                jax.grad(joint_logl, argnums=1), in_axes=(0, None)
            )(x, params)

            params = jax.tree_map(
                lambda x, y: x + self.h * jnp.mean(y, axis=0),
                params,
                grad_log_l_wrt_params,
            )

            return (x, params), None

        steps = jnp.arange(len(keys))  # needed to get a progress bar in scan

        fitted_params = jax.lax.scan(
            _step, init=(x_init, params_init), xs=(steps, keys)
        )[0]

        return fitted_params
