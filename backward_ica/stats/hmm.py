from jax import numpy as jnp, random, value_and_grad, tree_util, grad
from jax.tree_util import tree_leaves
from backward_ica.stats import LinearBackwardSmoother, State
from backward_ica.stats.kalman import Kalman
from backward_ica.stats.smc import SMC
from backward_ica.stats.distributions import * 
from backward_ica.stats.kernels import * 

from jax import lax, vmap
from backward_ica.utils import * 


from functools import partial
import optax

import copy 

def xtanh(slope):
    return lambda x: jnp.tanh(x) + slope*x



def get_generative_model(args, key_for_random_params=None):

    if args.model == 'linear':
        p = LinearGaussianHMM(args.state_dim, 
                                args.obs_dim, 
                                args.transition_matrix_conditionning, 
                                args.range_transition_map_params,
                                args.transition_bias, 
                                args.emission_bias)
    elif 'chaotic_rnn' in args.model or args.model == 'chaotic_rnn':
        if 'nonlinear_emission' in args.model:
            p = NonLinearHMM.chaotic_rnn_with_nonlinear_emission(args)
        else: 
            p = NonLinearHMM.chaotic_rnn(args)
    else: 
        p = NonLinearHMM.linear_transition_with_nonlinear_emission(args) # specify the structure of the true model
    
    if key_for_random_params is not None:
        theta_star = p.get_random_params(key_for_random_params, args)
        return p, theta_star
    else:
        return p


        
class HMM: 

    @register_pytree_node_class
    @dataclass(init=True)
    class Params:
        
        prior: Gaussian.NoiseParams 
        transition: Kernel.Params
        emission: Kernel.Params

        def compute_covs(self):
            self.prior.scale.cov
            self.transition.noise.scale.cov
            self.emission.noise.scale.cov

        def tree_flatten(self):
            return ((self.prior, self.transition, self.emission), None)

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            return cls(*children)

    parametrization = 'cov_chol'

    def __init__(self, 
                state_dim, 
                obs_dim, 
                transition_kernel_type, 
                emission_kernel_type,
                prior_dist=Gaussian):

        self.state_dim, self.obs_dim = state_dim, obs_dim
        self.prior_dist:Gaussian = prior_dist
        self.transition_kernel:Kernel = transition_kernel_type(state_dim)
        self.emission_kernel:Kernel = emission_kernel_type(state_dim, obs_dim)
        
    def sample_multiple_sequences(self, key, params, num_seqs, seq_length, single_split_seq=False, load_from='', loaded_seq=False):

        if loaded_seq:
            state_seq = jnp.load(os.path.join(load_from, 'x_data.npy')).astype(jnp.float64)
            obs_seq = jnp.load(os.path.join(load_from, 'y_data.npy')).astype(jnp.float64)
            
            print('Sequences loaded.')
            if single_split_seq: 
                return jnp.array(jnp.split(state_seq, num_seqs)), jnp.array(jnp.split(obs_seq, num_seqs))
            else: 
                return state_seq[:seq_length][jnp.newaxis,:], obs_seq[:seq_length][jnp.newaxis,:]
        else: 
            if single_split_seq: 
                state_seq, obs_seq = self.sample_seq(key, params, num_seqs*seq_length)
                return jnp.array(jnp.split(state_seq, num_seqs)), jnp.array(jnp.split(obs_seq, num_seqs))
            else: 
                key, *subkeys = random.split(key, num_seqs+1)
                sampler = vmap(self.sample_seq, in_axes=(0, None, None))
                return sampler(jnp.array(subkeys), params, seq_length)

    def get_random_params(self, key, params_to_set=None):
        key_prior, key_transition, key_emission = random.split(key, 3)

        prior_params = self.prior_dist.get_random_params(key_prior, 
                                                        self.state_dim)

        transition_params = self.transition_kernel.get_random_params(key_transition)
        emission_params = self.emission_kernel.get_random_params(key_emission)
        params = self.Params(prior_params, 
                        transition_params, 
                        emission_params)
        if params_to_set is not None: 
            params = self.set_params(params, params_to_set)
        return params 
        
    def format_params(self, params):

        return self.Params(self.prior_dist.format_params(params.prior),
                        self.transition_kernel.format_params(params.transition),
                        self.emission_kernel.format_params(params.emission))
                        
    def sample_seq(self, key, params, seq_length):

        params = self.format_params(params)

        keys = random.split(key, 2*seq_length)
        state_keys = keys[:seq_length]
        obs_keys = keys[seq_length:]

        prior_sample = self.prior_dist.sample(state_keys[0], params.prior)

        def _state_sample(carry, x):
            prev_sample = carry
            key = x
            sample = self.transition_kernel.sample(key, prev_sample, params.transition)
            return sample, sample
        _, state_seq = lax.scan(_state_sample, init=prior_sample, xs=state_keys[1:])

        state_seq = tree_prepend(prior_sample, state_seq)
        obs_seq = vmap(self.emission_kernel.sample, in_axes=(0,0,None))(obs_keys, state_seq, params.emission)

        return state_seq, obs_seq
        
    def print_num_params(self):
        params = self.get_random_params(random.PRNGKey(0))
        print('Num params:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves(params)))
        print('-- in prior + predict:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves((params.prior, params.transition))))
        print('-- in update:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves(params.emission)))

    def set_params(self, params, args):
        new_params = copy.deepcopy(params)
        for k,v in vars(args).items():         
            if k == 'default_prior_mean':
                new_params.prior.mean = v * jnp.ones_like(params.prior.mean)
            elif k == 'default_prior_base_scale':
                new_params.prior.scale = Scale.set_default(params.prior.scale, v, HMM.parametrization)
            elif k == 'default_transition_base_scale': 
                new_params.transition.noise.scale = Scale.set_default(params.transition.noise.scale, v, HMM.parametrization)
            elif k == 'default_emission_base_scale': 
                new_params.emission.noise.scale = Scale.set_default(params.emission.noise.scale, v, HMM.parametrization)
            elif k == 'default_emission_df':
                new_params.emission.noise.df = v
            elif k == 'default_emission_matrix' and hasattr(new_params.emission.map, 'w'):
                new_params.emission.map.w = v * jnp.ones_like(params.emission.map.w)
            elif (k == 'default_transition_matrix') and (self.transition_kernel.map_type != 'linear'):
                if (type(v) == str): new_params.transition.map['linear']['w'] = jnp.load(v).astype(jnp.float64)
        return new_params


class LinearGaussianHMM(HMM, LinearBackwardSmoother):

    def __init__(self, 
                state_dim, 
                obs_dim, 
                transition_matrix_conditionning=None,
                range_transition_map_params=(0,1),
                transition_bias=False,
                emission_bias=False):

        transition_kernel = Kernel.linear_gaussian(transition_matrix_conditionning, 
                                                    transition_bias, 
                                                    range_transition_map_params)
                                        
        emission_kernel = Kernel.linear_gaussian(None, emission_bias, (0,1))                     

        HMM.__init__(self, 
                    state_dim, 
                    obs_dim, 
                    transition_kernel_type = lambda state_dim: transition_kernel(state_dim, state_dim), 
                    emission_kernel_type = emission_kernel)

        LinearBackwardSmoother.__init__(self, state_dim)

    def empty_state(self):
        return State(out=(jnp.empty((self.state_dim,)), 
                        jnp.empty((self.state_dim, self.state_dim))), 
                    hidden=jnp.empty_like(jnp.empty((self.state_dim,))))

    def init_state(self, obs, params):

        mean, cov = Kalman.init(obs, params.prior, params.emission)

        return State(out=(mean, cov), hidden=jnp.empty_like(mean))
    
    def new_state(self, obs, prev_state, params):

        pred_mean, pred_cov = Kalman.predict(prev_state.out[0], prev_state.out[1], params.transition)
        
        mean, cov = Kalman.update(pred_mean, pred_cov, obs, params.emission)

        return State(out=(mean, cov), hidden=jnp.empty_like(mean))
    
    def backwd_params_from_state(self, state, params):

        return self.linear_gaussian_backwd_params_from_transition_and_filt(state.out[0], state.out[1], params.transition)

    def filt_params_from_state(self, state, params):
        return Gaussian.Params(mean=state.out[0], scale=Scale(cov=state.out[1]))

    def likelihood_seq(self, obs_seq, params):

        return Kalman.filter_seq(obs_seq, self.format_params(params))[-1]
    
    def fit_kalman_rmle(self, key, data, optimizer, learning_rate, batch_size, num_epochs, theta_star=None):
                
        
        key_init_params, key_batcher = random.split(key, 2)
        base_optimizer = getattr(optax, optimizer)(learning_rate)
        optimizer = base_optimizer
        # optimizer = optax.masked(base_optimizer, mask=HMM.Params(prior_mask, transition_mask, emission_mask))

        params = self.get_random_params(key_init_params)

        prior_scale = theta_star.prior.scale
        transition_scale = theta_star.transition.noise.scale
        emission_params = theta_star.emission

        def build_params(params):
            return HMM.Params(prior=Gaussian.Params(mean=params[0], scale=prior_scale), 
                            transition=Kernel.Params(params[1], transition_scale), 
                            emission=emission_params)
        
        params = (params.prior.mean, params.transition.map)

        loss = lambda seq, params: -self.likelihood_seq(seq, build_params(params))

        opt_state = optimizer.init(params)
        num_seqs = data.shape[0]
        
        @jit
        def batch_step(carry, x):
            
            def step(params, opt_state, batch):
                neg_logl_value, grads = vmap(value_and_grad(loss, argnums=1), in_axes=(0,None))(batch, params)
                avg_grads = tree_util.tree_map(jnp.mean, grads)
                updates, opt_state = optimizer.update(avg_grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return params, opt_state, -neg_logl_value.sum()

            data, params, opt_state = carry
            batch_start = x
            batch_obs_seq = lax.dynamic_slice_in_dim(data, batch_start, batch_size)
            params, opt_state, avg_logl_batch = step(params, opt_state, batch_obs_seq)
            return (data, params, opt_state), avg_logl_batch
        
        batch_start_indices = jnp.arange(0, num_seqs, batch_size)

        avg_logls = []
        all_params = []

        for epoch_nb in tqdm(range(num_epochs), 'Epoch'):

            key_batcher, subkey_batcher = random.split(key_batcher, 2)
            
            data = random.permutation(subkey_batcher, data)

            (_, params, opt_state), avg_logl_batches = lax.scan(batch_step, 
                                                                init=(data, params, opt_state), 
                                                                xs=batch_start_indices)
            avg_logls.append(avg_logl_batches.sum())
            all_params.append(params)

        best_optim = jnp.argmax(jnp.array(avg_logls))
        print(f'Best fit is epoch {best_optim} with logl {avg_logls[best_optim]}.')
        best_params = all_params[best_optim]
        
        return build_params(best_params), avg_logls, best_optim
    
    def compute_state_seq(self, obs_seq, compute_up_to, formatted_params):
        formatted_params.compute_covs()
        return super().compute_state_seq(obs_seq, compute_up_to, formatted_params)

class NonLinearHMM(HMM):

    @staticmethod
    def linear_transition_with_nonlinear_emission(args):
        if args.injective:
            nonlinear_map_forward = partial(Maps.neural_map, layers=args.emission_map_layers, slope=args.slope)
        else: 
            nonlinear_map_forward = partial(Maps.neural_map_noninjective, layers=args.emission_map_layers, slope=args.slope)
            
        transition_kernel_def = {'map':{'map_type':'linear',
                                        'map_info' : {'conditionning': args.transition_matrix_conditionning, 
                                        'bias': args.transition_bias,
                                        'range_params':args.range_transition_map_params}}, 
                                'noise_dist':Gaussian}


        emission_kernel_def = {'map':{'map_type':'nonlinear',
                                    'map_info' : {'homogeneous': True},
                                    'map': nonlinear_map_forward},
                            'noise_dist':Gaussian}


        return NonLinearHMM(args.state_dim, 
                            args.obs_dim, 
                            transition_kernel_def, 
                            emission_kernel_def, 
                            prior_dist = Gaussian,
                            num_particles = args.num_particles, 
                            num_smooth_particles=args.num_smooth_particles)
    @staticmethod
    def chaotic_rnn(args):
        nonlinear_map_forward = partial(Maps.chaotic_map, 
                                        grid_size=args.grid_size, 
                                        gamma=args.gamma,
                                        tau=args.tau)

        transition_kernel_def = {'map':{'map_type':'nonlinear',
                                        'map_info' : {'homogeneous': True},
                                        'map': nonlinear_map_forward},
                                'noise_dist':Gaussian}
        
        emission_kernel_def = {'map':{'map_type':'linear',
                                    'map_info' : {'conditionning': args.emission_matrix_conditionning, 
                                    'bias': args.emission_bias,
                                    'range_params':args.range_emission_map_params}}, 
                                'noise_dist':Student}

        return NonLinearHMM(args.state_dim, 
                            args.obs_dim, 
                            transition_kernel_def, 
                            emission_kernel_def, 
                            prior_dist = Gaussian,
                            num_particles = args.num_particles, 
                            num_smooth_particles=args.num_smooth_particles)
        
    @staticmethod
    def chaotic_rnn_with_nonlinear_emission(args):
        nonlinear_transition_map_forward = partial(Maps.chaotic_map, 
                                        grid_size=args.grid_size, 
                                        gamma=args.gamma,
                                        tau=args.tau)
        if args.injective:
            nonlinear_map_forward = partial(Maps.neural_map, layers=args.emission_map_layers, slope=args.slope)
        else: 
            nonlinear_map_forward = partial(Maps.neural_map_noninjective, layers=args.emission_map_layers, slope=args.slope)

        transition_kernel_def = {'map':{'map_type':'nonlinear',
                                        'map_info' : {'homogeneous': True},
                                        'map': nonlinear_transition_map_forward},
                                'noise_dist':Gaussian}
        
        emission_kernel_def = {'map':{'map_type':'nonlinear',
                                    'map_info' : {'homogeneous': True},
                                    'map': nonlinear_map_forward},
                            'noise_dist':Gaussian}

        return NonLinearHMM(args.state_dim, 
                            args.obs_dim, 
                            transition_kernel_def, 
                            emission_kernel_def, 
                            prior_dist = Gaussian,
                            num_particles = args.num_particles, 
                            num_smooth_particles=args.num_smooth_particles)
        
    def __init__(self, 
                state_dim, 
                obs_dim, 
                transition_kernel_def,
                emission_kernel_def,
                prior_dist = Gaussian,
                num_particles=100, 
                num_smooth_particles=None):
                                                
        HMM.__init__(self, 
                    state_dim, 
                    obs_dim, 
                    transition_kernel_type = lambda state_dim: Kernel(state_dim, state_dim, transition_kernel_def['map'], transition_kernel_def['noise_dist']), 
                    emission_kernel_type  = lambda state_dim, obs_dim:Kernel(state_dim, obs_dim, emission_kernel_def['map'], emission_kernel_def['noise_dist']),
                    prior_dist = prior_dist)

        self.smc = SMC(self.transition_kernel, 
                    self.emission_kernel, 
                    self.prior_dist, 
                    num_particles,
                    num_smooth_particles)

    def likelihood_seq(self, key, obs_seq, params):

        return self.smc.compute_filt_params_seq(key, 
                            obs_seq, 
                            self.format_params(params))[-1]
    
    def compute_filt_params_seq(self, key, obs_seq, formatted_params):

        return self.smc.compute_filt_params_seq(key, 
                                obs_seq, 
                                formatted_params)[0]

    def filt_seq(self, key, obs_seq, params):

        return self.compute_filt_params_seq(key, obs_seq, self.format_params(params))
        
    def compute_marginals(self, key, filt_seq, formatted_params):

        return self.smc.smooth_from_filt_seq(key, filt_seq, formatted_params)
    
    def smooth_seq(self, key, obs_seq, params):

        key, subkey = random.split(key, 2)

        formatted_params = self.format_params(params)

        filt_seq = self.smc.compute_filt_params_seq(key, 
                                obs_seq, 
                                formatted_params)[0]

        return self.smc.smooth_from_filt_seq(subkey, filt_seq, formatted_params)

    def filt_seq_to_mean_cov(self, key, obs_seq, params):

        weights, particles = self.filt_seq(key, obs_seq, params)
        means = vmap(lambda particles, weights: jnp.average(a=particles, axis=0, weights=weights))(particles, weights)
        covs = vmap(lambda mean, particles, weights: jnp.average(a=(particles-mean)**2, axis=0, weights=weights))(means, particles, weights)
        return means, covs 

    def smooth_seq_to_mean_cov(self, key, obs_seq, params):

        smoothing_paths = self.smooth_seq(key, obs_seq, params)
        return jnp.mean(smoothing_paths, axis=1), jnp.var(smoothing_paths, axis=1)

    def smooth_seq_at_multiple_timesteps(self, key, obs_seq, params, slices):
        key, subkey = random.split(key, 2)

        formatted_params = self.format_params(params)

        filt_seq = self.smc.compute_filt_params_seq(key, 
                                obs_seq, 
                                formatted_params)[0]

        def smooth_up_to_timestep(timestep):
            smoothing_paths = self.smc.smooth_from_filt_seq(subkey, tree_get_slice(0, timestep, filt_seq), formatted_params)
            return jnp.mean(smoothing_paths, axis=1), jnp.var(smoothing_paths, axis=1)
        means, covs = [], []

        for timestep in slices:
            mean, cov = smooth_up_to_timestep(timestep)
            means.append(mean)
            covs.append(cov)
            
        return means, covs

        
    def fit_ffbsi_em(self, key, data, optimizer, learning_rate, batch_size, num_epochs):

        key_init_params, key_batcher = random.split(key, 2)
        optimizer = getattr(optax, optimizer)(learning_rate)
        params = self.get_random_params(key_init_params)
        opt_state = optimizer.init(params)
        num_seqs = data.shape[0]        
        key_batcher, key_montecarlo = random.split(key, 2)
        mc_keys = random.split(key_montecarlo, num_seqs * num_epochs).reshape(num_epochs, num_seqs,-1)

        def e_from_smoothed_paths(theta, smoothed_paths, obs_seq):
            theta = self.format_params(theta)
            def _single_path_e_func(smoothed_path, obs_seq):
                init_val = self.prior_dist.logpdf(smoothed_path[0], theta.prior) \
                    + self.emission_kernel.logpdf(obs_seq[0], smoothed_path[0], theta.emission)
                def _step(prev_particle, particle, obs):
                    return self.transition_kernel.logpdf(particle, prev_particle, theta.transition) + \
                        self.emission_kernel.logpdf(obs, particle, theta.emission)
                return init_val + jnp.sum(vmap(_step)(smoothed_path[:-1], smoothed_path[1:], obs_seq[1:]))
            return jnp.mean(vmap(_single_path_e_func, in_axes=(0,None))(smoothed_paths, obs_seq))
        
        @jit
        def batch_step(carry, x):
            
            def step(prev_theta, opt_state, batch, keys):

                def e_step(key, obs_seq, theta):
                    
                    formatted_prev_theta = self.format_params(prev_theta)
                    
                    key_fwd, key_backwd = random.split(key, 2)
                    
                    filt_seq, logl_value = self.smc.compute_filt_params_seq(key_fwd, 
                            obs_seq, 
                            formatted_prev_theta)

                    smoothed_paths = self.smc.smooth_from_filt_seq(key_backwd, filt_seq, formatted_prev_theta)

                    return -e_from_smoothed_paths(theta, smoothed_paths, obs_seq), logl_value

                grads, logl_values = vmap(grad(e_step, argnums=2, has_aux=True), in_axes=(0,0, None))(keys, batch, prev_theta)
                avg_grads = tree_util.tree_map(jnp.mean, grads)
                updates, opt_state = optimizer.update(avg_grads, opt_state, prev_theta)
                theta = optax.apply_updates(prev_theta, updates)
                return theta, opt_state, jnp.mean(logl_values)

            data, params, opt_state, subkeys_epoch = carry
            batch_start = x
            batch_obs_seq = lax.dynamic_slice_in_dim(data, batch_start, batch_size)
            batch_keys = lax.dynamic_slice_in_dim(subkeys_epoch, batch_start, batch_size)
            params, opt_state, avg_logl_batch = step(params, opt_state, batch_obs_seq, batch_keys)
            return (data, params, opt_state, subkeys_epoch), avg_logl_batch
        
        batch_start_indices = jnp.arange(0, num_seqs, batch_size)

        avg_logls = []

        for epoch_nb in tqdm(range(num_epochs)):
            mc_keys_epoch = mc_keys[epoch_nb]
            key_batcher, subkey_batcher = random.split(key_batcher, 2)
            
            data = random.permutation(subkey_batcher, data)

            (_, params, opt_state, _), avg_logl_batches = lax.scan(batch_step, 
                                                                init=(data, params, opt_state, mc_keys_epoch), 
                                                                xs=batch_start_indices)
            avg_logls.append(jnp.mean(avg_logl_batches))

        
        return params, avg_logls
