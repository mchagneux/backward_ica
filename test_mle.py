import jax 
from backward_ica import hmm 

state_dim, obs_dim = 1,1
seq_length = 10 
p = hmm.LinearGaussianHMM(state_dim, obs_dim,
                        transition_matrix_conditionning='diagonal',
                        transition_bias=True,
                        emission_bias=False)

state_seqs, obs_seqs = p.sample_multiple