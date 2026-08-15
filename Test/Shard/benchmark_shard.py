"""Benchmark: chain×4 vs shard×4 across model complexity and dataset size.

Hypothesis: as compute-per-observation increases, the shard crossover point
(where shard×4 beats chain×4) shifts to smaller N.

Models
------
  simple   : mu = a + b*X             ~2 flops/obs
  poly20   : mu = X_poly(deg=20) @ beta  ~40 flops/obs
  scan100  : T=100 lax.scan tanh steps per obs  ~100 flops/obs

Only chain×4 vs shard×4 compared (same 4 chains, same 400 total samples).
shard×4 uses chain_method='vectorized' + auto-shard across 24 devices
→ 6 devices per chain, each chain's log-likelihood computed via XLA allreduce.
"""
import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from BayesForge import bf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PLATFORM      = "cpu"
CORES         = 24
NUM_CHAINS    = 4
TOTAL_SAMPLES = 400          # fixed across modes: 100/chain for both
WARMUP        = 200

N_SIMPLE  = [50_000, 200_000, 500_000, 1_000_000, 2_000_000]
N_POLY    = [10_000,  50_000, 200_000,   500_000, 1_000_000]
N_SCAN    = [ 2_000,  10_000,  50_000,   100_000,   200_000]

POLY_DEG  = 20
SCAN_T    = 100

# ---------------------------------------------------------------------------
# Single bf instance — device count fixed for the whole process
# ---------------------------------------------------------------------------
m = bf(platform=PLATFORM, cores=CORES, print_devices_found=True)
print(f"\nn_devices={m.n_devices}  chains={NUM_CHAINS}  "
      f"devices/chain (shard)={CORES // NUM_CHAINS}\n")

samples_per_chain = TOTAL_SAMPLES // NUM_CHAINS   # 100

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
def make_simple_model():
    def model(X, Y):
        alpha = m.dist.normal(0, 10)
        beta  = m.dist.normal(0, 10)
        sigma = m.dist.exponential(1)
        m.dist.normal(alpha + beta * X, sigma, obs=Y, name="Y")
    return model

def make_poly_model():
    def model(X, Y):
        # Build polynomial features via lax.scan to avoid XLA constant-folding blowup.
        # carry = current power of each obs; scan accumulates X, X^2, ..., X^POLY_DEG
        def build_powers(x):
            # start from 1.0 → yields [x, x², …, x^POLY_DEG]
            _, powers = lax.scan(
                lambda acc, _: (acc * x, acc * x),
                jnp.ones_like(x), None, length=POLY_DEG,
            )
            return powers  # (POLY_DEG,)

        Xp    = jax.vmap(build_powers)(X)   # (N, POLY_DEG)
        beta  = m.dist.normal(jnp.zeros(POLY_DEG), jnp.ones(POLY_DEG), name="beta")
        sigma = m.dist.exponential(1)
        m.dist.normal(Xp @ beta, sigma, obs=Y, name="Y")
    return model

def make_scan_model():
    def model(X, Y):
        w     = m.dist.normal(0, 1)
        sigma = m.dist.exponential(1)
        # Per observation: run SCAN_T tanh iterations (sequential, lax.scan)
        def process_one(x):
            final, _ = lax.scan(
                lambda c, _: (jnp.tanh(w * c + x), None),
                x, None, length=SCAN_T,
            )
            return final
        mu = jax.vmap(process_one)(X)
        m.dist.normal(mu, sigma, obs=Y, name="Y")
    return model

MODELS = {
    f"simple (2 flops/obs)":           (make_simple_model,  N_SIMPLE),
    f"poly deg={POLY_DEG} (~40 fl/obs)": (make_poly_model,   N_POLY),
    f"scan T={SCAN_T} (~100 fl/obs)":    (make_scan_model,   N_SCAN),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_data(N, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=N)
    Y = 2.0 + 0.8 * X + rng.normal(size=N)
    return jnp.array(X, dtype=jnp.float64), jnp.array(Y, dtype=jnp.float64)

def run(model_fn, X, Y, shard_flag):
    t0 = time.perf_counter()
    m.fit(model_fn(), obs=dict(X=X, Y=Y),
          num_samples=samples_per_chain, num_warmup=WARMUP,
          num_chains=NUM_CHAINS,
          shard=shard_flag,
          progress_bar=False)
    return time.perf_counter() - t0

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
all_results = {}

for model_name, (model_factory, N_range) in MODELS.items():
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'N':>10}  {'chain×4':>10}  {'shard×4':>10}  {'ratio':>8}")
    print("-" * 44)

    rows = []
    for N in N_range:
        X, Y = make_data(N)
        t_chain = run(model_factory, X, Y, shard_flag=False)
        t_shard = run(model_factory, X, Y, shard_flag=True)
        ratio   = t_chain / t_shard
        rows.append({"N": N, "chain": t_chain, "shard": t_shard, "ratio": ratio})
        print(f"{N:>10}  {t_chain:>9.2f}s  {t_shard:>9.2f}s  {ratio:>7.2f}x")

    all_results[model_name] = rows

# ---------------------------------------------------------------------------
# Plot — one subplot per model
# ---------------------------------------------------------------------------
n_models = len(MODELS)
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=False)

colors = {"chain": "#1f77b4", "shard": "#d62728"}

for ax, (model_name, rows) in zip(axes, all_results.items()):
    Ns      = [r["N"]     for r in rows]
    t_chain = [r["chain"] for r in rows]
    t_shard = [r["shard"] for r in rows]
    ratios  = [r["ratio"] for r in rows]

    ax.plot(Ns, t_chain, "o-", color=colors["chain"], label="chain×4")
    ax.plot(Ns, t_shard, "s-", color=colors["shard"], label="shard×4")

    # shade region where shard wins
    Ns_arr     = np.array(Ns)
    chain_arr  = np.array(t_chain)
    shard_arr  = np.array(t_shard)
    ax.fill_between(Ns_arr, chain_arr, shard_arr,
                    where=shard_arr < chain_arr,
                    alpha=0.15, color=colors["shard"], label="shard faster")

    # annotate speed-up ratio
    for r in rows:
        ax.annotate(f"{r['ratio']:.1f}×",
                    xy=(r["N"], min(r["chain"], r["shard"])),
                    xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=7,
                    color=colors["shard"] if r["ratio"] > 1 else colors["chain"])

    ax.set_xscale("log")
    ax.set_xlabel("N observations (log scale)")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(model_name, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    f"chain×4 vs shard×4  |  24 cores  |  {TOTAL_SAMPLES} total samples\n"
    f"shard: {NUM_CHAINS} chains × {CORES // NUM_CHAINS} devices/chain, "
    f"chain_method='vectorized' + XLA allreduce",
    fontsize=10,
)
plt.tight_layout()
plt.savefig("benchmark_shard.png", dpi=130)
print("\nPlot saved to benchmark_shard.png")
