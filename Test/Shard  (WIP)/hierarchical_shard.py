"""Hierarchical model — chain parallelism on CPU or GPU.

Varying-intercepts model run with multiple parallel chains.  Each chain
receives a full copy of the data; ``chain_method='parallel'`` (default)
dispatches one chain per JAX device via ``jax.pmap``.

Set PLATFORM='gpu' and CORES is ignored (physical GPUs are used).
"""
import numpy as np
import jax.numpy as jnp
from BayesForge import bf

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
N_groups = 12
N_per    = 200
N        = N_groups * N_per

true_mu_bar      = 5.0
true_sigma_group = 1.5
true_sigma_obs   = 0.5

group_means = rng.normal(true_mu_bar, true_sigma_group, size=N_groups)
group_id    = np.repeat(np.arange(N_groups), N_per)
Y_obs       = group_means[group_id] + rng.normal(scale=true_sigma_obs, size=N)

print(f"N={N}, groups={N_groups}, obs/group={N_per}")

# ---------------------------------------------------------------------------
# BayesForge setup
# ---------------------------------------------------------------------------
PLATFORM = "cpu"
CHAINS   = 4     # one per virtual device

m = bf(platform=PLATFORM, cores=CHAINS, print_devices_found=True)
print(f"n_devices : {m.n_devices}   n_chains : {CHAINS}")

# ---------------------------------------------------------------------------
# Model  (no sharding — each chain gets the full dataset)
# ---------------------------------------------------------------------------
def model(group_id, Y):
    mu_bar      = m.dist.normal(0, 10)
    sigma_group = m.dist.exponential(1)
    sigma_obs   = m.dist.exponential(1)
    alpha = m.dist.normal(
        jnp.ones(N_groups) * mu_bar, sigma_group,
        shape=(N_groups,), name="alpha",
    )
    m.dist.normal(alpha[group_id], sigma_obs, obs=Y, name="Y")

# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
m.fit(model,
      obs=dict(group_id=jnp.array(group_id), Y=jnp.array(Y_obs)),
      num_samples=500, num_warmup=500, num_chains=CHAINS,
      progress_bar=True)

print(m.summary())
print(f"\nTrue: mu_bar={true_mu_bar}, sigma_group={true_sigma_group}, "
      f"sigma_obs={true_sigma_obs}")
