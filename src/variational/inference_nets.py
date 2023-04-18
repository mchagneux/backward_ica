import haiku as hk 
from jax import numpy as jnp, nn 
from src.utils import *
from src.stats.distributions import * 
from src.stats.kernels import * 
from jax.flatten_util import ravel_pytree



def deep_gru(obs, prev_state, layers):

    gru = hk.DeepRNN([hk.GRU(hidden_size) for hidden_size in layers])

    return gru(obs, prev_state)

def gaussian_proj(state, d):

    net = hk.Linear(2*d,
        w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
        b_init=hk.initializers.RandomNormal())


    eta1, out2 = jnp.split(net(state.out),2)
    eta2 = -jnp.diag(nn.softplus(out2))# - jnp.eye(d)
    
    # eta1, out2 = out[:d], out[d:]
    # # eta2_chol = chol_from_vec(out2, d)

    # # eta2 = - (eta2_chol @ eta2_chol.T + jnp.eye(1))
    # eta2 = -jnp.diag(nn.relu(out2))

    return Gaussian.Params(
                    eta1=eta1, 
                    eta2=eta2)

def backwd_net(aux, obs, layers, state_dim):
    d = state_dim

    
    net = hk.nets.MLP((*layers, 2*d),
        w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
        b_init=hk.initializers.RandomNormal(),
        activate_final=False)
    
    eta1, out2 = jnp.split(net(obs),2)
    eta2 = -jnp.diag(nn.softplus(out2))# - jnp.eye(d)

    # eta_2_chol = jnp.diagonal(nn.softplus(out2))
    # eta2 = -(eta_2_chol @ eta_2_chol.T + jnp.eye(d))
    # eta2 = -jnp.diag(nn.softplus(out2))

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

def johnson(obs, layers, state_dim):


    d = state_dim 
    rec_net = hk.nets.MLP((*layers, 2*d),
                w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
                b_init=hk.initializers.RandomNormal(),
                activation=nn.tanh,
                activate_final=False)


    eta1, out2 = jnp.split(rec_net(obs), 2)
    eta2 = -jnp.diag(nn.softplus(out2))
    return eta1, eta2



# def linear_gaussian_proj(state, d):

#     A_back_dim = (d * (d+1)) // 2
#     a_back_dim = d
#     Sigma_back_dim = (d * (d + 1)) // 2
    
#     out_dim = A_back_dim + a_back_dim + Sigma_back_dim
    
#     net = hk.nets.MLP(output_sizes=(8,8,out_dim),
#                 activation=nn.tanh,
#                 activate_final=False, 
#                 w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
#                 b_init=hk.initializers.RandomNormal(),)

#     out = net(jnp.concatenate(state.hidden))

#     A_back_chol = chol_from_vec(out[:A_back_dim], d)
#     a_back = out[A_back_dim:A_back_dim+a_back_dim]
#     Sigma_back_vec = out[A_back_dim+a_back_dim:]

#     return Kernel.Params(map=Maps.LinearMapParams(w=A_back_chol @ A_back_chol.T + jnp.eye(d), b=a_back), 
#                         noise=Gaussian.NoiseParams.from_vec(Sigma_back_vec, d, chol_add=jnp.eye))