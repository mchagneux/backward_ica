import haiku as hk 
from jax import numpy as jnp, nn 
from backward_ica.utils import *
from backward_ica.stats.distributions import * 
from backward_ica.stats.kernels import * 

def deep_gru(obs, prev_state, layers):

    gru = hk.DeepRNN([hk.GRU(hidden_size) for hidden_size in layers])

    return gru(obs, prev_state)

def gaussian_proj(state, d):

    out_dim = d + (d * (d + 1)) // 2
    net = hk.Linear(out_dim, 
        w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
        b_init=hk.initializers.RandomNormal(),)

    out = net(state.out)
    
    return Gaussian.Params.from_vec(out, d, diag=False)

def backwd_update_forward(varying_params, next_state, layers, state_dim):

    d = state_dim
    out_dim = d + (d * (d+1)) // 2

    net = hk.nets.MLP((*layers, out_dim),
                w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
                b_init=hk.initializers.RandomNormal(),
                activation=nn.tanh,
                activate_final=False)
    
    out = net(jnp.concatenate((varying_params, next_state)))

    out = Gaussian.Params.from_vec(out, d, diag=False)

    return out.mean, out.scale

def johnson(obs, layers, state_dim):

    rec_net = hk.nets.MLP((*layers, 2*state_dim),
                w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
                b_init=hk.initializers.RandomNormal(),
                activation=nn.tanh,
                activate_final=False)


    out = rec_net(obs)
    eta1, log_prec_diag = jnp.split(out,2)

    eta2 = jnp.diag(nn.softplus(log_prec_diag))

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

def linear_gaussian_proj(state, d):

    A_back_dim = (d * (d+1)) // 2
    a_back_dim = d
    Sigma_back_dim = (d * (d + 1)) // 2
    
    out_dim = A_back_dim + a_back_dim + Sigma_back_dim
    net = hk.nets.MLP(output_sizes=(8,8,out_dim),
                activation=nn.tanh,
                activate_final=False, 
                w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
                b_init=hk.initializers.RandomNormal(),)

    out = net(jnp.concatenate(state.hidden))

    A_back_chol = chol_from_vec(out[:A_back_dim], d)
    a_back = out[A_back_dim:A_back_dim+a_back_dim]
    Sigma_back_vec = out[A_back_dim+a_back_dim:]

    return Kernel.Params(map=Maps.LinearMapParams(w=A_back_chol @ A_back_chol.T + jnp.eye(d), b=a_back), 
                        noise=Gaussian.NoiseParams.from_vec(Sigma_back_vec, d, chol_add=jnp.eye))