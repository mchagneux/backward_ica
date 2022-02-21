import jax.numpy as jnp
import jax
from numpy import true_divide

def _add(x,y):
    return x+y

def _identity(x,y):
    return x

def _step(carry, x):

    flag, y = x
    return jax.lax.cond(flag, _add, _identity, carry, y), None



flags = jnp.array([True, True, True, False, False, False])
ys = 2*jnp.ones_like(flags)

test_fun = lambda xs: jax.lax.scan(_step, init=0, xs=(flags,ys))[0]
fast_test_fun = jax.jit(test_fun)

print(fast_test_fun(flags))

