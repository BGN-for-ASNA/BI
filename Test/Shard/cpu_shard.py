"""CPU multi-device example: two distinct parallelism modes.

Mode A — Chain parallelism (chain_method='parallel', default)
-------------------------------------------------------------
* Each chain gets the FULL dataset replicated on its device.
* Best when num_chains == n_devices.
* pmap assigns one chain per device; no data split.

Mode B — Within-chain data parallelism (chain_method='vectorized')
------------------------------------------------------------------
* Data is sharded across ALL devices via NamedSharding.
* Chains are vectorized (vmap+jit), not pmapped.
* Inside each chain: alpha + beta * X is computed in parallel
  across shards; numpyro.sample calls jnp.sum on sharded
  log-probs → XLA allreduce → correct FULL log-likelihood.
* Allows num_chains < n_devices while still using all devices.
"""
import numpy as np
import jax.numpy as jnp
import time
from BayesForge import bf

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
N = 4_000
true_alpha, true_beta, true_sigma = 2.0, 0.7, 1.0

X = rng.normal(size=N).astype(np.float64)
Y = true_alpha + true_beta * X + rng.normal(scale=true_sigma, size=N)

# ===========================================================================
# MODE A: chain parallelism — 4 chains, 4 virtual devices
# ===========================================================================
CORES_A = 4
print("=" * 65)
print(f"Mode A  |  chain_method='parallel'  |  {CORES_A} chains, {CORES_A} devices")
print("=" * 65)

m_a = bf(platform="cpu", cores=CORES_A, print_devices_found=False)

def model_a(X, Y):
    alpha = m_a.dist.normal(0, 10)
    beta  = m_a.dist.normal(0, 10)
    sigma = m_a.dist.exponential(1)
    m_a.dist.normal(alpha + beta * X, sigma, obs=Y, name="Y")

t0 = time.perf_counter()
m_a.fit(model_a, obs=dict(X=jnp.array(X), Y=jnp.array(Y)),
        num_samples=500, num_warmup=500, num_chains=CORES_A,
        chain_method="parallel", progress_bar=False)
t_a = time.perf_counter() - t0
print(m_a.summary()[["mean", "sd"]])
print(f"Elapsed : {t_a:.1f}s  — full data per chain, {CORES_A} devices active\n")

# ===========================================================================
# MODE B: within-chain data parallelism — 4 chains, 8 devices
#
#  * alpha + beta * X  →  computed in parallel on all 8 shards
#  * jnp.sum(log_prob(Y_shard))  →  XLA allreduce  →  full log-likelihood
#  * posteriors identical to Mode A (all N observations, correct stats)
# ===========================================================================
CORES_B = 8   # more virtual devices than chains
N_CHAINS_B = 4
print("=" * 65)
print(f"Mode B  |  chain_method='vectorized'  |  "
      f"{N_CHAINS_B} chains, {CORES_B} devices")
print(f"        |  data sharded: N/8 per device, allreduce → full log-p")
print("=" * 65)

m_b = bf(platform="cpu", cores=CORES_B, print_devices_found=False)
# Data sharding is applied automatically inside fit() for chain_method='vectorized'

def model_b(X, Y):
    alpha = m_b.dist.normal(0, 10)
    beta  = m_b.dist.normal(0, 10)
    sigma = m_b.dist.exponential(1)
    # alpha + beta * X  computed in parallel across 8 device-shards
    m_b.dist.normal(alpha + beta * X, sigma, obs=Y, name="Y")

t0 = time.perf_counter()
m_b.fit(model_b, obs=dict(X=jnp.array(X), Y=jnp.array(Y)),
        num_samples=500, num_warmup=500, num_chains=N_CHAINS_B,
        chain_method="vectorized",   # <-- enables auto-sharding + allreduce
        progress_bar=False)
t_b = time.perf_counter() - t0
print(m_b.summary()[["mean", "sd"]])
print(f"Elapsed : {t_b:.1f}s  — {CORES_B} devices active, full posterior\n")

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
diff = (m_a.summary()["mean"] - m_b.summary()["mean"]).abs().max()
print("=" * 65)
print(f"True: alpha={true_alpha}, beta={true_beta}, sigma={true_sigma}")
print(f"Mode A elapsed : {t_a:.1f}s")
print(f"Mode B elapsed : {t_b:.1f}s")
print(f"Max mean diff  : {diff:.6f}  (≈0 → posteriors agree)")
print("=" * 65)
