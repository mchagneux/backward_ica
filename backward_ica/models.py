from jax import numpy as jnp, random, value_and_grad, tree_util, grad, config
from jax.tree_util import tree_leaves
from backward_ica.stats.kalman import Kalman
from backward_ica.stats.smc import SMC
from backward_ica.stats.distributions import * 
from backward_ica.stats.kernels import * 
import backward_ica.variational.inference_nets as inference_nets
from backward_ica.stats import LinearBackwardSmoother, TwoFilterSmoother

import haiku as hk
from jax import lax, vmap
from backward_ica.utils import * 


from functools import partial
from jax import nn
import optax

import copy 

def xtanh(slope):
    return lambda x: jnp.tanh(x) + slope*x



def get_generative_model(args, key_for_random_params=None):

    if args.p_version == 'linear':
        p = LinearGaussianHMM(args.state_dim, 
                                args.obs_dim, 
                                args.transition_matrix_conditionning, 
                                args.range_transition_map_params,
                                args.transition_bias, 
                                args.emission_bias)
    elif 'chaotic_rnn' in args.p_version:
        p = NonLinearHMM.chaotic_rnn(args)
    else: 
        p = NonLinearHMM.linear_transition_with_nonlinear_emission(args) # specify the structure of the true model
    
    if key_for_random_params is not None:
        theta_star = p.get_random_params(key_for_random_params, args)
        return p, theta_star
    else:
        return p


def get_variational_model(args, p=None, key_for_random_params=None):

    if args.q_version == 'linear':

        q = LinearGaussianHMM(state_dim=args.state_dim, 
                            obs_dim=args.obs_dim,
                            transition_matrix_conditionning=args.transition_matrix_conditionning,
                            range_transition_map_params=args.range_transition_map_params,
                            transition_bias=args.transition_bias, 
                            emission_bias=args.emission_bias)

    elif 'neural_backward_linear' in args.q_version:
        if (p is not None) and (p.transition_kernel.map_type == 'linear'):
            q = NeuralLinearBackwardSmoother.with_transition_from_p(p, args.update_layers)

        elif 'backwd_net' in args.q_version:
            q = NeuralLinearBackwardSmoother(state_dim=args.state_dim, 
                                                obs_dim=args.obs_dim,
                                                transition_kernel=None,
                                                update_layers=args.update_layers)
        else:
            q = NeuralLinearBackwardSmoother.with_linear_gaussian_transition_kernel(p, args.update_layers)
        
    # elif args.q_version == 'neural_backward':
    #     q = NeuralBackwardSmoother(state_dim=args.state_dim, 
    #                                     obs_dim=args.obs_dim, 
    #                                     update_layers=args.update_layers,
    #                                     backwd_layers=args.backwd_map_layers)

    elif args.q_version == 'johnson_backward':
            q = JohnsonBackward(state_dim=args.state_dim, 
                                    obs_dim=args.obs_dim, 
                                    layers=args.update_layers)

    elif args.q_version == 'johnson_forward':
            q = JohnsonForward(state_dim=args.state_dim, 
                                    obs_dim=args.obs_dim, 
                                    layers=args.update_layers)


    if key_for_random_params is not None:
        phi = q.get_random_params(key_for_random_params, args)
        return q, phi
    else:
        return q
        
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
        
    def sample_multiple_sequences(self, key, params, num_seqs, seq_length, single_split_seq=False, loaded_data=None):

        if loaded_data is not None: 
            state_seq, obs_seq = jnp.load(loaded_data[0]).astype(jnp.float64), jnp.load(loaded_data[1]).astype(jnp.float64)
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

State = namedtuple('State', ['out','hidden'])
GeneralBackwdState = namedtuple('BackwardState', ['inner', 'varying'])

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

    def init_state(self, obs, params):

        mean, cov = Kalman.init(obs, params.prior, params.emission)

        return State(out=(mean, cov), hidden=None)
    
    def new_state(self, obs, prev_state, params):

        pred_mean, pred_cov = Kalman.predict(prev_state.out[0], prev_state.out[1], params.transition)
        
        mean, cov = Kalman.update(pred_mean, pred_cov, obs, params.emission)

        return State(out=(mean, cov), hidden=None)
    
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
    
    def compute_state_seq(self, obs_seq, formatted_params):
        formatted_params.compute_covs()
        return super().compute_state_seq(obs_seq, formatted_params)

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

class NeuralLinearBackwardSmoother(LinearBackwardSmoother):

    @register_pytree_node_class
    @dataclass(init=True)
    class Params:

        prior:Any 
        state:Any 
        backwd:Any
        filt:Any

        def compute_covs(self):
            self.prior.scale.cov
            self.backwd.noise.scale.cov

        def tree_flatten(self):
            return ((self.prior, self.state, self.backwd, self.filt), None)

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            return cls(*children)

    @classmethod
    def with_transition_from_p(cls, p:HMM, layers=(8,8)):
        return cls(p.state_dim, 
                    p.obs_dim, 
                    p.transition_kernel, 
                    layers)
    
    @classmethod
    def with_linear_gaussian_transition_kernel(cls, p:HMM, layers):

        transition_kernel = hmm.Kernel.linear_gaussian(matrix_conditonning='init_invertible', 
                                                        bias=True, 
                                                        range_params=(-1,1))(p.state_dim, p.state_dim)
                                                        
        return cls(p.state_dim, p.obs_dim, transition_kernel, layers)


    def __init__(self, 
                state_dim,
                obs_dim, 
                transition_kernel=None,
                update_layers=(8,8)):
        

        super().__init__(state_dim)

        self.state_dim = state_dim
        self.obs_dim = obs_dim

        self.transition_kernel:Kernel = transition_kernel
        self.update_layers = update_layers
        d = self.state_dim
        
        self._state_net = hk.without_apply_rng(hk.transform(partial(inference_nets.deep_gru, 
                                                                    layers=self.update_layers)))
        self._filt_net = hk.without_apply_rng(hk.transform(partial(inference_nets.gaussian_proj, 
                                                                    d=d)))


        if self.transition_kernel is None:
            self._backwd_net = hk.without_apply_rng(hk.transform(partial(inference_nets.linear_gaussian_proj, d=d)))
            self._backwd_params_from_state = lambda state, params: self._backwd_net.apply(params.backwd, state)

        else: 
            def backwd_params_from_state(state, params):
                filt_params = self.filt_params_from_state(state, params)
                return NeuralLinearBackwardSmoother.linear_gaussian_backwd_params_from_transition_and_filt(filt_params.mean, 
                                                                                                        filt_params.scale.cov, 
                                                                                                        params.backwd)

            self._backwd_params_from_state = backwd_params_from_state
             
    def get_random_params(self, key, params_to_set=None):

        key_prior, key_state, key_filt, key_backwd = random.split(key, 4)

        dummy_obs = jnp.empty((self.obs_dim,))


        prior_params = tuple([random.normal(key, shape=[size]) for key, size in zip(random.split(key_prior, len(self.update_layers)), 
                                                                                    self.update_layers)])

        state_params = self._state_net.init(key_state, dummy_obs, prior_params)

        out, new_state = self._state_net.apply(state_params, dummy_obs, prior_params)

        dummy_state = State(out=out, 
                            hidden=new_state)

        filt_params = self._filt_net.init(key_filt, dummy_state)

        if self.transition_kernel is None:
            backwd_params = self._backwd_net.init(key_backwd, dummy_state)
        else: 
            backwd_params = self.transition_kernel.get_random_params(key_backwd)


        params =  self.Params(prior_params, 
                            state_params, 
                            backwd_params,
                            filt_params)
        
        if params_to_set is not None:
            params = self.set_params(params, params_to_set)
        return params  
        
    def set_params(self, params, args):
        new_params = copy.deepcopy(params)
        for k,v in vars(args).items():         
            if (k == 'default_transition_base_scale') and (self.transition_kernel is not None): 
                new_params.backwd.noise.scale = Scale.set_default(params.backwd.noise.scale, v, HMM.parametrization)
       
        return new_params

    def format_params(self, params):

        if self.transition_kernel is None:
            return params 
        else: 
            return self.Params(params.prior, 
                                params.state, 
                                self.transition_kernel.format_params(params.backwd), 
                                params.filt)

    def init_state(self, obs, params):
        out, init_state = self._state_net.apply(params.state, obs, params.prior)
        return State(out=out, hidden=init_state)

    def new_state(self, obs, prev_state, params):
        out, new_state = self._state_net.apply(params.state, obs, prev_state.hidden)
        return State(out=out, hidden=new_state)

    def filt_params_from_state(self, state, params):
        return self._filt_net.apply(params.filt, state)

    def backwd_params_from_state(self, state, params):
        return self._backwd_params_from_state(state, params)

    def print_num_params(self):
        params = self.get_random_params(random.PRNGKey(0))
        print('Num params:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves(params)))
        print('-- in prior:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves((params.prior,))))
        print('-- in state:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves((params.state,))))
        print('-- in backwd:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves((params.backwd,))))
        print('-- in filt:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves(params.filt)))
    


@register_pytree_node_class
@dataclass(init=True)
class JohnsonParams:
    prior: Gaussian.Params
    transition:Kernel.Params
    net:Any

    def compute_covs(self):
        self.prior.scale.cov
        self.transition.noise.scale.cov

    def tree_flatten(self):
        return ((self.prior, self.transition, self.net), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

class JohnsonSmoother:

    def __init__(self, state_dim, obs_dim, layers):

        self.state_dim = state_dim 
        self.obs_dim = obs_dim 
        self.prior_dist = Gaussian

        self.transition_kernel = Kernel.linear_gaussian(matrix_conditonning='diagonal',
                                                        bias=False, 
                                                        range_params=(-1,1))(state_dim, state_dim)

        self._net = hk.without_apply_rng(hk.transform(partial(inference_nets.johnson, layers=layers, state_dim=state_dim)))

    def get_random_params(self, key, params_to_set=None):
        key_prior, key_transition, key_net = random.split(key, 3)

        prior_params = self.prior_dist.get_random_params(key_prior, self.state_dim)
        transition_params = self.transition_kernel.get_random_params(key_transition)
        net_params = self._net.init(key_net, jnp.empty((self.obs_dim,)))

        params = JohnsonParams(prior_params, transition_params, net_params)
        if params_to_set is not None: 
            params = self.set_params(params, params_to_set)
        return params

    def format_params(self, params):
        return JohnsonParams(self.prior_dist.format_params(params.prior), 
                            self.transition_kernel.format_params(params.transition),
                            params.net)

    def set_params(self, params, args):
        new_params = copy.deepcopy(params)
        for k,v in vars(args).items():         
            if k == 'default_transition_base_scale': 
                new_params.transition.noise.scale = Scale.set_default(params.transition.noise.scale, v, HMM.parametrization)
            elif k == 'default_prior_base_scale':
                new_params.prior.scale = Scale.set_default(params.prior.scale, v, HMM.parametrization)
        return new_params

class JohnsonBackward(JohnsonSmoother, LinearBackwardSmoother):

    def __init__(self, state_dim, obs_dim, layers):

        JohnsonSmoother.__init__(self, state_dim, obs_dim, layers)
        LinearBackwardSmoother.__init__(self, state_dim)

    def init_state(self, obs, params):
        out = self._net.apply(params.net, obs)
        return Gaussian.Params.from_nat_params(out[0] + params.prior.eta1, out[1] + params.prior.eta2)

    def new_state(self, obs, prev_state, params):
        pred_mean, pred_cov = Kalman.predict(prev_state.mean, prev_state.scale.cov, params.transition)  

        pred = Gaussian.Params.from_mean_cov(pred_mean, pred_cov)
        out = self._net.apply(params.net, obs)

        return Gaussian.Params.from_nat_params(out[0] + pred.eta1, out[1] + pred.eta2)

    def filt_params_from_state(self, state, params):
        return state

    def backwd_params_from_state(self, state, params):
        return self.linear_gaussian_backwd_params_from_transition_and_filt(state.mean, state.scale.cov, params.transition)

    def compute_state_seq(self, obs_seq, formatted_params):
        formatted_params.compute_covs()
        return super().compute_state_seq(obs_seq, formatted_params)

BackwdVar = namedtuple('BackwdVar', ['base', 'tilde'])

class JohnsonForward(JohnsonSmoother, TwoFilterSmoother):
    
    @staticmethod
    def linear_gaussian_forward_params_from_backwd_variable_and_transition(backwd_variable_tilde:Gaussian.Params, 
                                                                            transition_params:Kernel.Params):
        A, R_prec = transition_params.map.w, transition_params.noise.scale.prec

        eta1, eta2 = backwd_variable_tilde.eta1, backwd_variable_tilde.eta2
        prec_forward = R_prec + eta2

        K = inv(prec_forward)

        A_forward = K @ R_prec @ A
        b_forward = K @ eta1

        return Kernel.Params(map=Maps.LinearMapParams(A_forward, b_forward), 
                            noise=Gaussian.NoiseParams(Scale(prec=prec_forward)))
        
    def __init__(self, state_dim, obs_dim, layers):
        JohnsonSmoother.__init__(self, state_dim, obs_dim, layers)
        
        TwoFilterSmoother.__init__(self, state_dim, 
                                    forward_kernel=Kernel.linear_gaussian(matrix_conditonning=None, 
                                                                        bias=True, 
                                                                        range_params=(0,1)))

    def init_filt_params(self, state, params):
        return Gaussian.Params.from_nat_params(state[0] + params.prior.eta1, state[1] + params.prior.eta2)

    def new_filt_params(self, state, prev_filt_params, params):
        pred_mean, pred_cov = Kalman.predict(prev_filt_params.mean, prev_filt_params.scale.cov, params.transition)  

        pred = Gaussian.Params.from_mean_cov(pred_mean, pred_cov)

        return Gaussian.Params.from_nat_params(state[0] + pred.eta1, state[1] + pred.eta2)
        
    def init_backwd_var(self, state, params):


        d = self.state_dim 
        base = Gaussian.Params.from_nat_params(eta1=jnp.zeros((d,)), 
                                               eta2=jnp.zeros((d,d)))               


        return BackwdVar(base=base, tilde=Gaussian.Params.from_nat_params(*state))

    def compute_state(self, obs, params):
        return self._net.apply(params.net, obs)

    def new_backwd_var(self, state, next_backwd_var, params):

        next_eta1_tilde, next_eta2_tilde = next_backwd_var.tilde.eta1, next_backwd_var.tilde.eta2

        A, R = params.transition.map.w, params.transition.noise.scale.cov
        K = inv(jnp.eye(self.state_dim) + next_eta2_tilde @ R)

        base = Gaussian.Params.from_nat_params(eta1 = A.T @ K @ next_eta1_tilde, 
                                                eta2 = A.T @ K @ next_eta2_tilde @ A)


        tilde = Gaussian.Params.from_nat_params(state[0] + base.eta1, 
                                                state[1] + base.eta2)


        return BackwdVar(base=base,
                        tilde=tilde)

    def forward_params_from_backwd_var(self, backwd_var:BackwdVar, params):
        return self.linear_gaussian_forward_params_from_backwd_variable_and_transition(backwd_var.tilde, params.transition)

    def compute_state_seq(self, obs_seq, formatted_params):
        formatted_params.compute_covs()
        return vmap(self.compute_state, in_axes=(0,None))(obs_seq, formatted_params)

    def compute_filt_params_seq(self, state_seq, formatted_params):

        init_filt_params = self.init_filt_params(tree_get_idx(0,state_seq), 
                                                formatted_params)

        @jit
        def _step(carry, state):
            prev_filt_params, formatted_params = carry
            filt_params = self.new_filt_params(state, prev_filt_params, formatted_params)
            return (filt_params, formatted_params), filt_params

        filt_params_seq = lax.scan(_step, 
                            init=(init_filt_params, formatted_params), 
                            xs=tree_dropfirst(state_seq))[1]

        return tree_prepend(init_filt_params, filt_params_seq)

    def compute_backwd_variables_seq(self, state_seq, formatted_params):

        last_backwd_var = self.init_backwd_var(tree_get_idx(-1, state_seq), formatted_params)

        @jit
        def _step(carry, state):
            next_backwd_var, formatted_params = carry 
            backwd_var = self.new_backwd_var(state, next_backwd_var, formatted_params)
            return (backwd_var, formatted_params), backwd_var

        backwd_variables_seq = lax.scan(_step, 
                                        init=(last_backwd_var, formatted_params),
                                        xs=tree_droplast(state_seq), 
                                        reverse=True)[1]

        return tree_append(backwd_variables_seq, last_backwd_var)

    def compute_marginal(self, filt_params:Gaussian.Params, backwd_variable:Gaussian.Params):
        mu, Sigma = filt_params.mean, filt_params.scale.cov
        kappa, Pi = backwd_variable.base.eta1, backwd_variable.base.eta2
        K = Sigma @ inv(jnp.eye(self.state_dim) + Pi @ Sigma)
        marginal_mean = mu + K @ (kappa - Pi @ mu)
        marginal_cov = Sigma - K @ Pi @ Sigma
        return Gaussian.Params(mean=marginal_mean, scale=Scale(cov=marginal_cov))

    def compute_marginals(self, filt_params_seq, backwd_variables_seq):
        
        return vmap(self.compute_marginal)(filt_params_seq, backwd_variables_seq)

    def smooth_seq(self, obs_seq, params, lag=None):
        
        formatted_params = self.format_params(params)
        formatted_params.compute_covs()

        state_seq = self.compute_state_seq(obs_seq, formatted_params)
        marginal_smoothing_stats =  self.compute_marginals(self.compute_filt_params_seq(state_seq, formatted_params),
                                                            self.compute_backwd_variables_seq(state_seq, formatted_params))

        return marginal_smoothing_stats.mean, marginal_smoothing_stats.scale.cov

    def smooth_seq_at_multiple_timesteps(self, obs_seq, params, slices):
        formatted_params = self.format_params(params)
        formatted_params.compute_covs()
        state_seq = self.compute_state_seq(obs_seq, formatted_params)
        filt_params_seq = self.compute_filt_params_seq(state_seq, formatted_params)


        def smooth_up_to_timestep(timestep):
            marginals = self.compute_marginals(filt_params_seq=tree_get_slice(0, timestep, filt_params_seq), 
                                                backwd_variables_seq=self.compute_backwd_variables_seq(tree_get_slice(0, timestep, state_seq), formatted_params))
            return marginals.mean, marginals.scale.cov
        means, covs = [], []

        for timestep in slices:
            mean, cov = smooth_up_to_timestep(timestep)
            means.append(mean)
            covs.append(cov)
            
        return means, covs  

    def filt_seq(self, obs_seq, params):
        formatted_params = self.format_params(params)
        formatted_params.compute_covs()
        
        state_seq = self.compute_state_seq(obs_seq, formatted_params)
        filt_params_seq =  self.compute_filt_params_seq(state_seq, formatted_params)

        return vmap(lambda x:x.mean)(filt_params_seq), vmap(lambda x:x.scale.cov)(filt_params_seq)
        




    
# class NeuralBackwardSmoother(BackwardSmoother):

#     def __init__(self, 
#                 state_dim, 
#                 obs_dim,
#                 update_layers,
#                 backwd_layers,
#                 filt_dist=Gaussian):

#         self.state_dim = state_dim 
#         self.obs_dim = obs_dim 

#         self.update_layers = update_layers

#         self.filt_params_shape = jnp.sum(jnp.array(update_layers))

#         self.filt_update_init_params, self.filt_update_apply = hk.without_apply_rng(hk.transform(partial(self.filt_update_forward, 
#                                                                 layers=update_layers, 
#                                                                 state_dim=state_dim)))
        

#         backwd_kernel_map_def = {'map_type':'nonlinear',
#                                 'map_info' : {'homogeneous': False, 'varying_params_shape':self.filt_params_shape},
#                                 'map': partial(self.backwd_update_forward, layers=backwd_layers)}
                

#         super().__init__(filt_dist, 
#                         Kernel(state_dim, state_dim, backwd_kernel_map_def, Gaussian))

#     def get_random_params(self, key, args=None):
        
#         key_prior, key_filt, key_back = random.split(key, 3)
    
#         dummy_obs = jnp.ones((self.obs_dim,))


#         key_priors = random.split(key_prior, len(self.update_layers))
#         prior_params = tuple([random.normal(key, shape=[size]) for key, size in zip(key_priors, self.update_layers)])
#         filt_update_params = self.filt_update_init_params(key_filt, dummy_obs, prior_params)

#         backwd_params = self.backwd_kernel.get_random_params(key_back)

#         return GeneralBackwardSmootherParams(prior=prior_params,
#                                             filt_update=filt_update_params, 
#                                             backwd=backwd_params)

#     def smooth_seq(self, key, obs_seq, params, num_samples, lag=None):

#         formatted_params = self.format_params(params)

#         filt_params_seq = self.compute_filt_params_seq(obs_seq, formatted_params)
#         backwd_params_seq = self.compute_backwd_params_seq(filt_params_seq, formatted_params)

#         if lag is None:
#             marginals = self.compute_marginals(key, filt_params_seq, backwd_params_seq, num_samples)
#         else: 
#             marginals = self.compute_fixed_lag_marginals(key, filt_params_seq, backwd_params_seq, num_samples, lag)

#         return jnp.mean(marginals, axis=0), jnp.var(marginals, axis=0)

#     def format_params(self, params):
#         return params

#     def init_filt_params(self, obs, params):
#         return FiltState(*self.filt_update_apply(params.filt_update, obs, params.prior))

#     def new_filt_params(self, obs, filt_params:FiltState, params):
#         return FiltState(*self.filt_update_apply(params.filt_update, obs, filt_params.hidden))

#     def get_init_state(self):
#         return tuple([jnp.zeros(shape=[size]) for size in self.update_layers])

#     def new_backwd_params(self, filt_params:FiltState, params):

#         return BackwardState(params.backwd, jnp.concatenate(filt_params.hidden))

#     def compute_marginals(self, key, filt_params_seq, backwd_params_seq, num_samples):

#         def _sample_for_marginals(key, last_filt_params:FiltState, backwd_params_seq):
            
#             keys = random.split(key, backwd_params_seq.varying.shape[0]+1)

#             last_sample = self.filt_dist.sample(keys[-1], last_filt_params.out)

#             def _sample_step(next_sample, x):
                
#                 key, backwd_params = x
#                 sample = self.backwd_kernel.sample(key, next_sample, backwd_params)
#                 return sample, sample
            
#             samples = lax.scan(_sample_step, init=last_sample, xs=(keys[:-1], backwd_params_seq), reverse=True)[1]

#             return tree_append(samples, last_sample)

#         parallel_sampler = jit(vmap(_sample_for_marginals, in_axes=(0,None,None)))

#         return parallel_sampler(random.split(key, num_samples), tree_get_idx(-1, filt_params_seq), backwd_params_seq)

#     def compute_fixed_lag_marginals(self, key, filt_params_seq, backwd_params_seq, num_samples, lag):
        
#         def _sample_for_marginals(key, filt_params_seq, backwd_params_seq):

#             def _sample_for_marginal(init, x):

#                 key, lagged_filt_params, strided_backwd_params_subseq = x

#                 keys = random.split(key, strided_backwd_params_subseq.varying.shape[0]+1)

#                 last_sample = self.filt_dist.sample(keys[-1], lagged_filt_params.out)

#                 def _sample_step(next_sample, x):
                    
#                     key, backwd_params = x
#                     sample = self.backwd_kernel.sample(key, next_sample, backwd_params)
#                     return sample, None

#                 marginal_sample = lax.scan(_sample_step, 
#                                         init=last_sample, 
#                                         xs=(keys[:-1], strided_backwd_params_subseq), 
#                                         reverse=True)[0]

#                 return None, marginal_sample
                

#             return lax.scan(_sample_for_marginal, 
#                                 init=None, 
#                                 xs=(random.split(key, backwd_params_seq.varying.shape[0]-lag+1), tree_get_slice(lag, None, filt_params_seq), tree_get_strides(lag, backwd_params_seq)))[1]
        
#         parallel_sampler = jit(vmap(_sample_for_marginals, in_axes=(0,None,None)))

#         return parallel_sampler(random.split(key, num_samples), filt_params_seq, backwd_params_seq)
    
#     def print_num_params(self):
#         params = self.get_random_params(random.PRNGKey(0))
#         print('Num params:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves(params)))
#         print('-- filt net:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves((params.filt_update))))
#         print('-- prior state:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves((params.prior))))
#         print('-- backwd net:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves((params.backwd))))
        


            
        



        

