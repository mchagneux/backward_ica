from src.stats.hmm import LinearGaussianHMM
from src.stats.ula import ULA
import jax

p = LinearGaussianHMM(
    state_dim=10,
    obs_dim=10,
    transition_matrix_conditionning="diagonal",
    range_transition_map_params=(0.9, 0.98),
)

key = jax.random.PRNGKey(0)

theta = p.get_random_params(key)

true_x, y = p.sample_seq(key, theta, 1_000)


ula = ULA(p, num_steps=1_000, h=1e-3, num_particles=50)
# Perform smoothing
x_smoothed = ula.smooth(key, y, theta)

import matplotlib.pyplot as plt

state_dim = true_x.shape[1]
fig = plt.figure(figsize=(12, 4 * state_dim))
gs = fig.add_gridspec(state_dim, 1)

for d in range(state_dim):
    # Smoothing plot
    ax = fig.add_subplot(gs[d, 0])
    ax.plot(true_x[:, d], label="True", linewidth=2)
    ax.plot(x_smoothed[:, d], label="Smoothed", linewidth=2, alpha=0.8)
    ax.set_title(f"Smoothing - Dimension {d+1}", pad=10, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)

plt.tight_layout(h_pad=3.0)
plt.show()
