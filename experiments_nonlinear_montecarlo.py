#%%
from backward_ica.elbo import NonLinearELBO
import backward_ica.hmm as hmm
import jax 
import jax.numpy as jnp
import haiku as hk 
import optax 
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key,2)
state_dim, obs_dim = 2,3 
seq_length = 16 
num_seqs = 32
import matplotlib.pyplot as plt


p_params, p_def = hmm.get_random_params(subkey, 
                                    state_dim, 
                                    obs_dim,
                                    transition_mapping_type='linear',
                                    emission_mapping_type='nonlinear')

p = hmm.GaussianHMM.build_from_dict(p_params, p_def)

key, *subkeys = jax.random.split(key, num_seqs+1)
state_seqs, obs_seqs = jax.vmap(p.sample, in_axes=(0, None))(jnp.array(subkeys), seq_length)

#%%

def backwd(filt_mean, filt_cov):

    net = hk.nets.MLP((32, 3*state_dim))
    out = net(jnp.concatenate((filt_mean, jnp.diagonal(filt_cov))))
    A = jnp.diag(out[:state_dim])
    a = out[state_dim:2*state_dim]
    log_cov_diag = out[2*state_dim:]
    return A, a, jnp.diag(jnp.exp(log_cov_diag))


def filt(obs, filt_mean, filt_cov):
    net = hk.nets.MLP((32, 2*state_dim))
    out = net(jnp.concatenate((obs, filt_mean, jnp.diagonal(filt_cov))))
    mean = out[:state_dim]
    log_cov_diag = out[state_dim:]
    return mean, jnp.diag(jnp.exp(log_cov_diag))

key, *subkeys = jax.random.split(key, 3)

filt_init, filt_apply = hk.without_apply_rng(hk.transform(filt))
backward_init, backward_apply = hk.without_apply_rng(hk.transform(backwd))

dummy_obs = obs_seqs[0][0]
dummy_mean = jnp.empty((state_dim,))
dummy_cov = jnp.empty((state_dim, state_dim))

filt_params = filt_init(subkeys[0], dummy_obs, dummy_mean, dummy_cov)
backward_params = backward_init(subkeys[1], dummy_mean, dummy_cov)


q_def = {'filtering':filt_apply, 
        'backward':backward_apply}

q_params = {'filtering':filt_params,
            'backward':backward_params}
#%% 

elbo = NonLinearELBO(p_def, q_def).compute
loss = lambda obs_seq, key, q_params: -elbo(obs_seq, key, p_params, q_params)[0]

optimizer = optax.adam(learning_rate=1e-3)

batch_size = 8
num_batches = num_seqs // batch_size
     
@jax.jit
def q_step(q_params, opt_state, batch, keys):
    loss_values, grads = jax.vmap(jax.value_and_grad(loss, argnums=2), in_axes=(0,0,None))(batch, keys, q_params)
    avg_loss_value = jnp.mean(loss_values)
    avg_grads = jax.tree_util.tree_map(jnp.mean, grads)
    updates, opt_state = optimizer.update(avg_grads, opt_state, q_params)
    q_params = optax.apply_updates(q_params, updates)
    return q_params, opt_state, avg_loss_value

num_epochs = 1000
key, *subkeys = jax.random.split(key, num_seqs * num_epochs + 1)
subkeys = jnp.array(subkeys)
def fit(q_params):
    opt_state = optimizer.init(q_params)
    avg_neg_elbos = []

    def loader(obs_seqs, keys):
        for index in range(0, seq_length, batch_size):
            yield obs_seqs[index:index + batch_size], keys[num_seqs*loader.epoch_cnt + index:num_seqs*loader.epoch_cnt + index + batch_size]
        loader.epoch_cnt +=1
    loader.epoch_cnt = 0

    for _ in range(num_epochs):
        avg_neg_elbo_epoch = 0.0
        for (batch, keys) in loader(obs_seqs, subkeys):
            q_params, opt_state, avg_neg_elbo = q_step(q_params, opt_state, batch, keys)
            avg_neg_elbo_epoch += avg_neg_elbo / num_batches
        avg_neg_elbos.append(avg_neg_elbo_epoch)

    return q_params, avg_neg_elbos

fitted_q_params, avg_neg_elbos = fit(q_params)
key, *subkeys = jax.random.split(key, num_seqs+1)
elbos, (marginal_means, marginal_covs) = jax.vmap(elbo, in_axes=(0,0,None,None))(obs_seqs, jnp.array(subkeys), p_params, fitted_q_params)

print('Expectation under q_phi against true states:',jnp.mean((marginal_means - state_seqs)**2))

plt.plot(avg_neg_elbos)
plt.xlabel('Epoch nb') 
plt.ylabel('$-\mathcal{L}(\\theta,\\phi)$')
plt.savefig('train')