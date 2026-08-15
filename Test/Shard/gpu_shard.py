"""GPU sharding example — Bayesian linear regression distributed across GPUs.

Each GPU receives a disjoint slice of the observation arrays; XLA fuses the
per-device log-likelihood contributions automatically.  Chain parallelism
(one MCMC chain per GPU) stacks on top of the data parallelism for maximum
throughput.

Run on a multi-GPU machine:
    python gpu_shard.py
"""
import numpy as np
import jax.numpy as jnp
import jax
import time
from BayesForge import bf

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
N = 20_000
true_alpha, true_beta, true_sigma = 3.0, -1.5, 0.8

X = rng.normal(size=N).astype(np.float64)
Y = true_alpha + true_beta * X + rng.normal(scale=true_sigma, size=N)

# ---------------------------------------------------------------------------
# Setup — detect GPUs automatically
# ---------------------------------------------------------------------------
m = bf(platform="gpu", print_devices_found=True)

if m.n_gpus == 0:
    raise SystemExit("No GPUs found. Run on a machine with at least one GPU.")

print(f"\nGPUs available : {m.n_gpus}")
print(f"Device mesh    : {m.mesh}")

# ---------------------------------------------------------------------------
# Shard observations across all GPUs
# Each GPU will own N // n_gpus consecutive rows of X and Y.
# ---------------------------------------------------------------------------
X_sharded = m.shard(jnp.array(X))
Y_sharded = m.shard(jnp.array(Y))

print(f"\nX sharding: {X_sharded.sharding}")
print(f"Y sharding: {Y_sharded.sharding}")

# ---------------------------------------------------------------------------
# Model — identical to the single-GPU version; no sharding-aware code needed
# ---------------------------------------------------------------------------
def model(X, Y):
    alpha = m.dist.normal(0, 10)
    beta  = m.dist.normal(0, 10)
    sigma = m.dist.exponential(1)
    m.dist.normal(alpha + beta * X, sigma, obs=Y, name="Y")

# ---------------------------------------------------------------------------
# Fit — chain_method='parallel' (BayesForge default) sends each chain to
# a separate GPU via jax.pmap, while the sharded data provides within-chain
# data parallelism.  Together: n_gpus chains × n_gpus data shards.
# ---------------------------------------------------------------------------
print(f"\nFitting with {m.n_gpus} chains (one per GPU) + sharded data …")
t0 = time.perf_counter()
m.fit(model,
      obs=dict(X=X_sharded, Y=Y_sharded),
      num_samples=1000,
      num_warmup=1000,
      num_chains=m.n_gpus,
      progress_bar=True)
elapsed = time.perf_counter() - t0

print(f"\nElapsed: {elapsed:.1f}s")
print(m.summary())
print(f"\nTrue: alpha={true_alpha}, beta={true_beta}, sigma={true_sigma}")
