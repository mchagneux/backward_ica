import jax
import optax
from jax import vmap, lax, config, numpy as jnp
from jax.random import normal
config.update("jax_enable_x64", True)

from .hmm import *
from .utils import *


def get_keys(key, num_seqs, num_epochs):
    keys = jax.random.split(key, num_seqs * num_epochs)
    keys = jnp.array(keys).reshape(num_epochs, num_seqs,-1)
    return keys

def get_dummy_keys(key, num_seqs, num_epochs): 
    return jnp.empty((num_epochs, num_seqs, 1))

def init_rep_net_forward(state_dim, out_dim):
    d = state_dim
    eta1_num_params = d 
    eta2_num_params = (d * (d+1)) // 2 
    def rep_net_forward(x):
        net = hk.nets.MLP((8, eta1_num_params + eta2_num_params))
        out = net(x)
        eta1 = out[:eta1_num_params]
        eta2 = jnp.zeros((d,d)).at[jnp.tril_indices(d)].set(out[eta1_num_params:])
        return eta1, - eta2 @ eta2.T
    dummy_obs = jnp.empty((out_dim,))
    return hk.without_apply_rng(hk.transform(rep_net_forward(dummy_obs)))

def constant_terms_from_log_gaussian(dim:int, log_det:float)->float:
    """Utility function to compute the log of the term that is against the exponential for a multivariate Normal

    Args:
        dim (int): the dimension of the support of the multivariate Normal
        det (float): the precomputed determinant of the covariance matrix 

    Returns:
        float: the value of the requested factor  
    """

    return -0.5*(dim * jnp.log(2*jnp.pi) + log_det)

def transition_term_integrated_under_backward(q_backwd_state, transition_params):
    # expectation of the quadratic form that appears in the log of the state transition density

    A = transition_params.map.w @ q_backwd_state.map.w - jnp.eye(transition_params.scale.cov.shape[0])
    b = transition_params.map.w @ q_backwd_state.map.b + transition_params.map.b
    Omega = transition_params.scale.prec
    
    result = -0.5 * QuadTerm.from_A_b_Omega(A, b, Omega)
    result.c += -0.5 * jnp.trace(transition_params.scale.prec @ transition_params.map.w @ q_backwd_state.scale.cov @ transition_params.map.w.T) \
                + constant_terms_from_log_gaussian(transition_params.scale.cov.shape[0], transition_params.scale.log_det)
    return result 

def expect_quadratic_term_under_backward(quad_form:QuadTerm, backwd_state):
    # the result is still a quadratic forms with new parameters, following the formula for expected values of quadratic forms  

    W = backwd_state.map.w.T @ quad_form.W @ backwd_state.map.w
    v = backwd_state.map.w.T @ (quad_form.v + (quad_form.W + quad_form.W.T) @ backwd_state.map.b)
    c = quad_form.c + jnp.trace(quad_form.W @ backwd_state.scale.cov) + backwd_state.map.b.T @ quad_form.W @ backwd_state.map.b + quad_form.v.T @ backwd_state.map.b 

    return QuadTerm(W=W, v=v, c=c)

def expect_quadratic_term_under_gaussian(quad_form:QuadTerm, gaussian_params):
    return jnp.trace(quad_form.W @ gaussian_params.scale.cov) + quad_form.evaluate(gaussian_params.mean)

def quadratic_term_from_log_gaussian(gaussian_params):

    result = - 0.5 * QuadTerm(W=gaussian_params.scale.prec, 
                    v=-(gaussian_params.scale.prec + gaussian_params.scale.prec.T) @ gaussian_params.mean, 
                    c=gaussian_params.mean.T @ gaussian_params.scale.prec @ gaussian_params.mean)

    result.c += constant_terms_from_log_gaussian(gaussian_params.mean.shape[0], gaussian_params.scale.log_det)

    return result

def get_tractable_emission_term(obs, emission_params):
    A = emission_params.map.w
    b = emission_params.map.b - obs
    Omega = emission_params.scale.prec
    emission_term = -0.5*QuadTerm.from_A_b_Omega(A, b, Omega)
    emission_term.c += constant_terms_from_log_gaussian(emission_params.scale.cov.shape[0], emission_params.scale.log_det)
    return emission_term

def get_tractable_emission_term_from_natparams(emission_natparams):
    eta1, eta2 = emission_natparams
    const = -0.25 * eta1.T @ jnp.linalg.solve(eta2, eta1) - 0.5 * jnp.log(jnp.linalg.det(-2*eta2)) - eta1.shape[0] * jnp.log(jnp.pi)
    return QuadTerm(W=eta2, 
                    v=eta1, 
                    c=const)



class GeneralELBO:

    def __init__(self, p:HMM, q:BackwardSmoother, num_samples=200):

        self.p = p
        self.q = q
        self.num_samples = num_samples

    def __call__(self, key, obs_seq, theta:HMMParams, phi):

        phi.compute_covs()

        filt_state_seq = self.q.compute_filt_seq(obs_seq, phi)
        backwd_state_seq = self.q.compute_backwd_seq(filt_state_seq, phi)

        def _monte_carlo_sample(key, obs_seq, last_filt_state, backwd_state_seq):

            keys = jax.random.split(key, obs_seq.shape[0])
            last_sample = self.q.filt_dist.sample(keys[-1], last_filt_state)

            last_term = -self.q.filt_dist.logpdf(last_sample, last_filt_state) \
                    + self.p.emission_kernel.logpdf(obs_seq[-1], last_sample, theta.emission)

            def _sample_step(next_sample, x):
                
                key, obs, backwd_state = x
                sample = self.q.backwd_kernel.sample(key, next_sample, backwd_state)

                emission_term_p = self.p.emission_kernel.logpdf(obs, sample, theta.emission)
                transition_term_p = self.p.transition_kernel.logpdf(next_sample, sample, theta.transition)
                backwd_term_q = -self.q.backwd_kernel.logpdf(sample, next_sample, backwd_state)

                return sample, emission_term_p + transition_term_p + backwd_term_q
            
            init_sample, terms = lax.scan(_sample_step, init=last_sample, xs=(keys[:-1], obs_seq[:-1], backwd_state_seq), reverse=True)

            return self.p.prior_dist.logpdf(init_sample, theta.prior) + jnp.sum(terms) + last_term

        parallel_sampler = vmap(_monte_carlo_sample, in_axes=(0,None,None,None))

        keys = jax.random.split(key, self.num_samples)
        last_filt_state =  tree_get_idx(-1, filt_state_seq)
        mc_samples = parallel_sampler(keys, obs_seq, last_filt_state, backwd_state_seq)
        return jnp.mean(mc_samples)


class BackwardLinearTowerELBO:

    def __init__(self, p:HMM, q:LinearBackwardSmoother, num_samples=200):
        self.p = p
        self.q = q
        self.num_samples = num_samples

    def __call__(self, key, obs_seq, theta:HMMParams, phi):


        def compute_kl_term(obs_seq):

            kl_term = quadratic_term_from_log_gaussian(theta.prior)

            q_filt_state = self.q.init_filt_state(obs_seq[0], phi)

            def V_step(state, obs):

                q_filt_state, kl_term = state
                q_backwd_state = self.q.new_backwd_state(q_filt_state, phi)

                kl_term = expect_quadratic_term_under_backward(kl_term, q_backwd_state) \
                        + transition_term_integrated_under_backward(q_backwd_state, theta.transition)


                kl_term.c += -constant_terms_from_log_gaussian(self.p.state_dim, q_backwd_state.scale.log_det) +  0.5 * self.p.state_dim
                q_filt_state = self.q.new_filt_state(obs, q_filt_state, phi)

                return (q_filt_state, kl_term), q_backwd_state
        
            (q_last_filt_state, kl_term), q_backwd_state_seq = lax.scan(V_step, 
                                                            init=(q_filt_state, kl_term), 
                                                            xs=obs_seq[1:])


            kl_term = expect_quadratic_term_under_gaussian(kl_term, q_last_filt_state) \
                        - constant_terms_from_log_gaussian(self.p.state_dim, q_last_filt_state.scale.log_det) \
                        + 0.5*self.p.state_dim

            return kl_term, (q_last_filt_state, q_backwd_state_seq)

        kl_term, (q_last_filt_state, q_backwd_state_seq) = compute_kl_term(obs_seq)

        marginals = self.q.backwd_pass(q_last_filt_state, q_backwd_state_seq)
        
        def sample_from_marginal(normal_sample, marginal, obs, theta):
            common_term = obs - self.p.emission_kernel.map(marginal.mean + marginal.scale.chol @ normal_sample, theta.emission).squeeze()
            return -0.5 * (common_term.T @ theta.emission.scale.prec @ common_term)


        # def _sample_step(carry, x):
        #     next_state_sample = carry
        #     matrix, bias, cov, obs, normal_sample = x
        #     current_state_sample = matrix @ next_state_sample + bias + jnp.linalg.cholesky(cov) @ normal_sample
        #     common_term = obs - self.p.emission_kernel.map(jnp.atleast_2d(current_state_sample), theta.emission).squeeze()
        #     return current_state_sample, -0.5 * (common_term.T @ theta.emission.scale.prec @ common_term)
            
        # matrices = jnp.concatenate((q_backwd_state_seq.map.w, jnp.zeros((1,self.p.state_dim, self.p.state_dim))))
        # biases = jnp.concatenate((q_backwd_state_seq.map.b, q_last_filt_state.mean[None,:]))
        # covs = jnp.concatenate((q_backwd_state_seq.scale.cov, q_last_filt_state.scale.cov[None,:]))

        # sample_path = lambda normal_samples_seq: lax.scan(_sample_step, 
        #                                                 init=jnp.empty((self.p.state_dim,)), 
        #                                                 xs=(matrices, biases, covs, obs_seq, normal_samples_seq), 
        #                                                 reverse=True)[1]

        normal_samples = normal(key, shape=(self.num_samples, obs_seq.shape[0], self.p.state_dim))

        monte_carlo_samples = vmap(vmap(sample_from_marginal, in_axes=(0,0,0,None)), in_axes=(0,None,None,None))(normal_samples, marginals, obs_seq, theta)
        
        reconstruction_term = jnp.sum(jnp.mean(monte_carlo_samples, axis=0)) + \
             obs_seq.shape[0] * constant_terms_from_log_gaussian(theta.emission.scale.cov.shape[0], theta.emission.scale.log_det)

        # reconstruction_term = jnp.sum(jnp.mean(jax.vmap(sample_path)(normal_samples), axis=0)) \
        #                     + obs_seq.shape[0] * constant_terms_from_log_gaussian(theta.emission.scale.cov.shape[0], theta.emission.scale.log_det)

                        
        return reconstruction_term + kl_term



class LinearGaussianTowerELBO:

    def __init__(self, p:HMM, q:LinearGaussianHMM):
        self.p = p
        self.q = q
        
    def __call__(self, obs_seq, theta:HMMParams, phi:HMMParams):

        result = quadratic_term_from_log_gaussian(theta.prior) + get_tractable_emission_term(obs_seq[0], theta.emission)


        q_filt_state = self.q.init_filt_state(obs_seq[0], phi)

        def V_step(state, obs):

            q_filt_state, kl_term = state
            q_backwd_state = self.q.new_backwd_state(q_filt_state, phi)

            kl_term = expect_quadratic_term_under_backward(kl_term, q_backwd_state) \
                    + transition_term_integrated_under_backward(q_backwd_state, theta.transition) \
                    + get_tractable_emission_term(obs, theta.emission)


            kl_term.c += -constant_terms_from_log_gaussian(self.p.state_dim, q_backwd_state.scale.log_det) +  0.5 * self.p.state_dim
            q_filt_state = self.q.new_filt_state(obs, q_filt_state, phi)

            return (q_filt_state, kl_term), q_backwd_state
    
        (q_last_filt_state, result) = lax.scan(V_step, 
                                                init=(q_filt_state, result), 
                                                xs=obs_seq[1:])[0]


        return expect_quadratic_term_under_gaussian(result, q_last_filt_state) \
                    - constant_terms_from_log_gaussian(self.p.state_dim, q_last_filt_state.scale.log_det) \
                    + 0.5*self.p.state_dim
    
class JohnsonTowerELBO:

    def __init__(self, p:HMM, q:BackwardSmoother, aux_map, num_samples=200):

        self.p = p
        self.q = q
        self.aux_map = aux_map
        self.num_samples = num_samples

    def __call__(self, obs_seq, theta:HMMParams, phi, *args):

        key, aux_params = args

        def tractable_terms(obs_seq):

            kl_term = quadratic_term_from_log_gaussian(theta.prior) 
            aux_reconstruction_term  = get_tractable_emission_term_from_natparams(self.aux_map(aux_params, obs_seq[0]))

            q_filt_state = self.q.init_filt_state(obs_seq[0], phi)

            def V_step(state, obs):

                q_filt_state, kl_term, aux_reconstruction_term = state
                q_backwd_state = self.q.new_backwd_state(q_filt_state, phi)

                kl_term = expect_quadratic_term_under_backward(kl_term, q_backwd_state) \
                        + transition_term_integrated_under_backward(q_backwd_state, theta.transition)

                aux_reconstruction_term = expect_quadratic_term_under_backward(aux_reconstruction_term, q_backwd_state) \
                        + get_tractable_emission_term_from_natparams(self.aux_map(aux_params, obs))


                kl_term.c += -constant_terms_from_log_gaussian(self.p.state_dim, q_backwd_state.scale.log_det) +  0.5 * self.p.state_dim
                q_filt_state = self.q.new_filt_state(obs, q_filt_state, phi)

                return (q_filt_state, kl_term, aux_reconstruction_term), q_backwd_state
        
            (q_last_filt_state, kl_term, aux_reconstruction_term), q_backwd_state_seq = lax.scan(V_step, 
                                                            init=(q_filt_state, kl_term, aux_reconstruction_term), 
                                                            xs=obs_seq[1:])


            kl_term = expect_quadratic_term_under_gaussian(kl_term, q_last_filt_state) \
                        - constant_terms_from_log_gaussian(self.p.state_dim, q_last_filt_state.scale.log_det) \
                        + 0.5*self.p.state_dim
            aux_reconstruction_term = expect_quadratic_term_under_gaussian(aux_reconstruction_term, q_last_filt_state)

            return (kl_term, aux_reconstruction_term), (q_last_filt_state, q_backwd_state_seq)

        (kl_term, aux_reconstruction_term), (q_last_filt_state, q_backwd_state_seq) = tractable_terms(obs_seq)

        normal_samples = normal(key, shape=(self.num_samples, obs_seq.shape[0], self.p.state_dim))

        def _sample_step(carry, x):
            next_state_sample = carry
            matrix, bias, cov, obs, normal_sample = x
            current_state_sample = matrix @ next_state_sample + bias + jnp.linalg.cholesky(cov) @ normal_sample
            common_term = obs - self.p.emission_kernel.map(jnp.atleast_2d(current_state_sample), theta.emission).squeeze()
            return current_state_sample, -0.5 * (common_term.T @ theta.emission.scale.prec @ common_term)
            
        matrices = jnp.concatenate((q_backwd_state_seq.map.w, jnp.zeros((1,self.p.state_dim, self.p.state_dim))))
        biases = jnp.concatenate((q_backwd_state_seq.map.b, q_last_filt_state.mean[None,:]))
        covs = jnp.concatenate((q_backwd_state_seq.scale.cov, q_last_filt_state.scale.cov[None,:]))

        sample_path = lambda normal_samples_seq: lax.scan(_sample_step, 
                                                        init=jnp.empty((self.p.state_dim,)), 
                                                        xs=(matrices, biases, covs, obs_seq, normal_samples_seq), 
                                                        reverse=True)[1]

        reconstruction_term = jnp.sum(jnp.mean(jax.vmap(sample_path)(normal_samples), axis=0)) \
                            + obs_seq.shape[0] * constant_terms_from_log_gaussian(theta.emission.scale.cov.shape[0], theta.emission.scale.log_det)

                        
        return kl_term + aux_reconstruction_term, kl_term + reconstruction_term

class SVITrainer:

    def __init__(self, p:HMM, q:BackwardSmoother, optimizer, learning_rate, num_epochs, batch_size, num_samples=1, force_mc=False):


        # schedule = lambda num_batches: optax.piecewise_constant_schedule(learning_rate, {150 * num_batches:0.1})
        # schedule_fn = optax.piecewise_constant_schedule(1., {100*: decay_rate})
        # self.optimizer = optax.chain(optimizer(learning_rate), optax.scale_by_schedule(schedule_fn))
        self.optimizer = lambda num_batches: getattr(optax, optimizer)(learning_rate)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.q = q 
        self.q.print_num_params()
        self.p = p 

        if not force_mc: 
            if isinstance(self.p, LinearGaussianHMM):
                self.elbo = LinearGaussianTowerELBO(self.p, self.q)
                self.get_montecarlo_keys = get_dummy_keys
            else: 
                self.elbo = BackwardLinearTowerELBO(self.p, self.q, num_samples)
                self.get_montecarlo_keys = get_keys
        else:
            self.elbo = GeneralELBO(self.p, self.q, num_samples)
            self.get_montecarlo_keys = get_keys


        
    def fit(self, key_batcher, key_montecarlo, data, theta, phi, store_every):

        if isinstance(self.elbo, LinearGaussianTowerELBO):
            loss = lambda key, seq, phi: -self.elbo(seq, self.p.format_params(theta), self.q.format_params(phi))
        else:
            loss = lambda key, seq, phi: -self.elbo(key, seq, self.p.format_params(theta), self.q.format_params(phi))
        params = phi

        num_seqs = data.shape[0]
        optimizer = self.optimizer(num_seqs // self.batch_size)

        opt_state = optimizer.init(params)
        subkeys = self.get_montecarlo_keys(key_montecarlo, num_seqs, self.num_epochs)


        @jax.jit
        def batch_step(carry, x):

            def step(params, opt_state, batch, keys):
                neg_elbo_values, grads = jax.vmap(jax.value_and_grad(loss, argnums=2), in_axes=(0,0,None))(keys, batch, params)
                avg_grads = jax.tree_util.tree_map(jnp.mean, grads)
                updates, opt_state = optimizer.update(avg_grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return params, opt_state, jnp.mean(-neg_elbo_values)

            data, params, opt_state, subkeys_epoch = carry
            batch_start = x
            batch_obs_seq = jax.lax.dynamic_slice_in_dim(data, batch_start, self.batch_size)
            batch_keys = jax.lax.dynamic_slice_in_dim(subkeys_epoch, batch_start, self.batch_size)
            params, opt_state, avg_elbo_batch = step(params, opt_state, batch_obs_seq, batch_keys)
            return (data, params, opt_state, subkeys_epoch), avg_elbo_batch


        avg_elbos = []
        all_params = dict()
        batch_start_indices = jnp.arange(0, num_seqs, self.batch_size)

        for epoch_nb in range(self.num_epochs):
            subkeys_epoch = subkeys[epoch_nb]
            key_batcher, subkey_batcher = jax.random.split(key_batcher, 2)
            
            data = jax.random.permutation(subkey_batcher, data)
        
            (_ , params, opt_state, _), avg_elbo_batches = jax.lax.scan(batch_step,  
                                                                init=(data, params, opt_state, subkeys_epoch), 
                                                                xs = batch_start_indices)
            if epoch_nb % store_every == 0:
                all_params[epoch_nb] = params

            avg_elbos.append(jnp.mean(avg_elbo_batches))
        
        all_params[epoch_nb] = params
                    
        return all_params, avg_elbos

    def profile(self, key, data, theta):

        params_key, monte_carlo_key = jax.random.split(key, 2)
        phi = self.q.get_random_params(params_key)

        if self.use_johnson: 
            aux_params = self.aux_init_params(params_key, data[0][0])
        else:
            aux_params = None        
        if self.use_johnson: 
            loss = lambda seq, key, phi, aux_params: self.loss(seq, key, self.p.format_params(theta), self.q.format_params(phi), aux_params)
            params = (phi, aux_params)
        else: 
            loss = lambda seq, key, phi: self.loss(seq, key, self.p.format_params(theta), self.q.format_params(phi), None)
            params = phi

        num_seqs = data.shape[0]
        subkeys = self.get_montecarlo_keys(monte_carlo_key, num_seqs, 1).squeeze()

        @jit
        def step(params, batch, keys):
            return jax.vmap(loss, in_axes=(0,0,None))(batch, keys, params)

        step(params, data[:self.batch_size], subkeys[:self.batch_size])

        with jax.profiler.trace('./profiling/'):
            print(step(params, data[:2], subkeys[:2]))


    def check_elbo(self, data, theta):
        if isinstance(self.p, LinearGaussianHMM):
            print('Checking ELBO quality...')

            avg_evidences = vmap(jit(lambda seq: self.p.likelihood_seq(seq, theta)))(data)
            theta = self.p.format_params(theta)
            if isinstance(self.elbo, LinearGaussianTowerELBO):
                elbo = jit(lambda key, seq:self.elbo(seq, theta, theta))
            else: 
                elbo = jit(lambda key, seq: self.elbo(key, seq, theta, theta))
            keys = jax.random.split(jax.random.PRNGKey(0), data.shape[0])
            avg_elbos = vmap(elbo)(keys, data)
            print('Avg error with Kalman evidence:', jnp.mean(jnp.abs(avg_evidences-avg_elbos)))

        

    def multi_fit(self, key_params, key_batcher, key_montecarlo, data, theta, num_fits, store_every=None):

        self.check_elbo(data, theta)

        if store_every is None: 
            store_every = self.num_epochs

        all_avg_elbos = []
        all_params = []
        print('-- Starting training...')
        for fit_nb, subkey_params in enumerate(jax.random.split(key_params, num_fits)):
            key_batcher, subkey_batcher = jax.random.split(key_batcher, 2)
            key_montecarlo, subkey_montecarlo = jax.random.split(key_montecarlo, 2)
            params, avg_elbos = self.fit(subkey_batcher, subkey_montecarlo, data, theta, self.q.get_random_params(subkey_params), store_every)
            all_avg_elbos.append(avg_elbos)
            all_params.append(params)

            print(f'End of fit {fit_nb+1}/{num_fits}, final ELBO {avg_elbos[-1]:.3f}')


        array_to_sort = np.array([avg_elbos[-1] for avg_elbos in all_avg_elbos])
        array_to_sort[np.isnan(array_to_sort)] = -np.inf
        best_optim = jnp.argmax(array_to_sort)
        print(f'Best fit is {best_optim+1}.')
        best_params = all_params[best_optim]
        return best_params, (best_optim, list(best_params.keys()), all_avg_elbos)



def check_linear_gaussian_elbo(p:LinearGaussianHMM, num_seqs, seq_length):
    key_params, key_gen = jax.random.split(jax.random.PRNGKey(0), 2)
    theta = p.get_random_params(key_params)

    seqs = p.sample_multiple_sequences(key_gen, theta, num_seqs, seq_length)[1]

    elbo = LinearGaussianTowerELBO(p,p)

    evidence_reference = vmap(jit(lambda seq: p.likelihood_seq(seq, theta)))(seqs)
    theta = p.format_params(theta)
    evidence_elbo = vmap(jit(lambda seq: elbo(seq, theta, theta)))(seqs)

    print('ELBO sanity check:',jnp.mean(jnp.abs(evidence_elbo - evidence_reference)))

def check_general_elbo(p:LinearGaussianHMM, num_seqs, seq_length, num_samples):

    key_params, key_gen = jax.random.split(jax.random.PRNGKey(0), 2)
    theta = p.get_random_params(key_params)

    seqs = p.sample_multiple_sequences(key_gen, theta, num_seqs, seq_length)[1]
    mc_keys = jax.random.split(jax.random.PRNGKey(2), num_seqs)
    elbo = GeneralELBO(p,p,num_samples)

    evidence_reference = vmap(jit(lambda seq: p.likelihood_seq(seq, theta)))(seqs)
    
    theta = p.format_params(theta)
    evidence_elbo = vmap(jit(lambda key, seq: elbo(key, seq, theta, theta)))(mc_keys, seqs)

    print('ELBO sanity check:',jnp.mean(jnp.abs(evidence_elbo - evidence_reference)))


