import jax.random as random 
import jax.numpy as jnp
import matplotlib.pyplot as plt
from backward_ica.hmm import init_nica_params, nica_mlp
key = random.PRNGKey(0)
import haiku as hk
from functools import partial
s_dim = 1
x_dim = 1

s = jnp.ones((s_dim,))




class NicaMLPModule(hk.Module):
    def __init__(self, obs_dim, layers):
        super(NicaMLPModule, self).__init__()        
        self.init = lambda shape, dtype: init_nica_params(hk.next_rng_key(), shape[0], layers, obs_dim)

    def __call__(self, s):
        params = hk.get_parameter(name='w', shape=s.shape, init=self.init)
        return nica_mlp(params, s)

def nica_mlp_func(s, obs_dim, layers):
    net = NicaMLPModule(obs_dim, layers)
    return net(s)



init, apply = hk.without_apply_rng(hk.transform(partial(nica_mlp_func, obs_dim=x_dim, layers=5)))

print(init(random.PRNGKey(0), s))




# def test(s, obs_dim, layers):
#     net = nica_mlp_hk_module(s, obs_dim, layers)




# params = init(random.PRNGKey(0), s)
# print(params)
# x = 
# plt.plot(s, x)
# plt.savefig('test')