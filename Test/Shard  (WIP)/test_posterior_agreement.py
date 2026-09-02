"""
Posterior agreement: chain×4 vs shard×4 across three model types.

For each model both methods are fitted with the same number of total samples.
KDE densities are overlaid per parameter and symmetric KL divergence is
reported.  Small KL (< 0.01 nats) = posteriors are statistically equivalent.

Models
------
  simple : alpha + beta*X + eps          (params: alpha, beta, sigma)
  poly5  : sum_k beta_k * X^k + eps      (params: beta[0..4], sigma)
  scan20 : T=20 lax.scan tanh per obs    (params: w, sigma)
"""
import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from BayesForge import bf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PLATFORM   = "cpu"
CORES      = 24
NUM_CHAINS = 4
N_SAMPLES  = 500    # per chain → 2000 total

N_SIMPLE = 200_000 ; WARMUP_SIMPLE = 300
N_POLY   =  20_000 ; POLY_DEG = 5 ; WARMUP_POLY = 1_000
N_SCAN   =   5_000 ; SCAN_T   = 20 ; WARMUP_SCAN = 300

# ---------------------------------------------------------------------------
m = bf(platform=PLATFORM, cores=CORES, print_devices_found=False)
print(f"n_devices={m.n_devices}   chains={NUM_CHAINS}   "
      f"total_samples={NUM_CHAINS * N_SAMPLES}\n")

# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------
def _rng(seed=0): return np.random.default_rng(seed)

def make_simple(N, seed=0):
    rng = _rng(seed)
    X = rng.normal(size=N)
    Y = 2.0 + 0.8 * X + rng.normal(size=N)
    return dict(X=jnp.array(X), Y=jnp.array(Y))

def make_poly(N, seed=0):
    rng = _rng(seed)
    X = rng.normal(size=N)
    true_beta = np.array([0.8, -0.5, 0.3, -0.2, 0.1])
    Y = sum(true_beta[k] * X**(k+1) for k in range(POLY_DEG)) + rng.normal(size=N)
    return dict(X=jnp.array(X), Y=jnp.array(Y))

def make_scan(N, seed=0):
    rng = _rng(seed)
    X = rng.normal(size=N)
    def apply(x):
        c = x
        for _ in range(SCAN_T):
            c = np.tanh(0.5 * c + x)
        return c
    Y = np.vectorize(apply)(X) + rng.normal(size=N)
    return dict(X=jnp.array(X), Y=jnp.array(Y))

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
def simple_model(X, Y):
    alpha = m.dist.normal(0, 10)
    beta  = m.dist.normal(0, 10)
    sigma = m.dist.exponential(1)
    m.dist.normal(alpha + beta * X, sigma, obs=Y, name="Y")

def poly_model(X, Y):
    def build_powers(x):
        # start from 1.0 → yields [x, x², x³, x⁴, x⁵]
        _, powers = lax.scan(
            lambda acc, _: (acc * x, acc * x),
            jnp.ones_like(x), None, length=POLY_DEG,
        )
        return powers
    Xp    = jax.vmap(build_powers)(X)
    beta  = m.dist.normal(jnp.zeros(POLY_DEG), jnp.ones(POLY_DEG), name="beta")
    sigma = m.dist.exponential(1)
    m.dist.normal(Xp @ beta, sigma, obs=Y, name="Y")

def scan_model(X, Y):
    w     = m.dist.normal(0, 1)
    sigma = m.dist.exponential(1)
    def process(x):
        final, _ = lax.scan(
            lambda c, _: (jnp.tanh(w * c + x), None),
            x, None, length=SCAN_T,
        )
        return final
    mu = jax.vmap(process)(X)
    m.dist.normal(mu, sigma, obs=Y, name="Y")

# ---------------------------------------------------------------------------
# Fit helper
# ---------------------------------------------------------------------------
def fit_both(model_fn, obs, warmup):
    print("  fitting chain×4 ...")
    m.fit(model_fn, obs=obs,
          num_samples=N_SAMPLES, num_warmup=warmup,
          num_chains=NUM_CHAINS, shard=False, progress_bar=False)
    chain = {k: np.array(v)
             for k, v in m.sampler.get_samples(group_by_chain=False).items()}

    print("  fitting shard×4 ...")
    m.fit(model_fn, obs=obs,
          num_samples=N_SAMPLES, num_warmup=warmup,
          num_chains=NUM_CHAINS, shard=True, progress_bar=False)
    shard = {k: np.array(v)
             for k, v in m.sampler.get_samples(group_by_chain=False).items()}

    return chain, shard

# ---------------------------------------------------------------------------
# KL divergence (symmetric)
# ---------------------------------------------------------------------------
def sym_kl(a, b, n_grid=600):
    lo = min(a.min(), b.min()) - abs(a.std())
    hi = max(a.max(), b.max()) + abs(a.std())
    x  = np.linspace(lo, hi, n_grid)
    p  = np.clip(gaussian_kde(a)(x), 1e-12, None)
    q  = np.clip(gaussian_kde(b)(x), 1e-12, None)
    p /= p.sum(); q /= q.sum()
    return 0.5 * (np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p)))

# ---------------------------------------------------------------------------
# Expand vectorised params (beta → beta[0], beta[1], ...)
# ---------------------------------------------------------------------------
def flatten_params(posteriors):
    out = {}
    for k, v in posteriors.items():
        v = np.array(v)
        if v.ndim == 1:
            out[k] = v
        else:
            for i in range(v.shape[1]):
                out[f"{k}[{i}]"] = v[:, i]
    return out

# ---------------------------------------------------------------------------
# Run all three models
# ---------------------------------------------------------------------------
MODELS = [
    ("simple  N=200k", simple_model, make_simple(N_SIMPLE), WARMUP_SIMPLE),
    ("poly5   N=20k",  poly_model,   make_poly(N_POLY),     WARMUP_POLY),
    ("scan20  N=5k",   scan_model,   make_scan(N_SCAN),     WARMUP_SCAN),
]

all_results = []
for label, model_fn, obs, warmup in MODELS:
    print(f"\n{'='*55}\nModel: {label}  (warmup={warmup})")
    chain, shard = fit_both(model_fn, obs, warmup)
    chain_flat   = flatten_params(chain)
    shard_flat   = flatten_params(shard)
    params = sorted(chain_flat.keys())
    kls = {p: sym_kl(chain_flat[p], shard_flat[p]) for p in params}
    all_results.append(dict(label=label, chain=chain_flat,
                             shard=shard_flat, params=params, kls=kls))
    print(f"  params: {params}")
    for p, kl in kls.items():
        print(f"    KL({p}) = {kl:.6f} nats")

# ---------------------------------------------------------------------------
# Plot — one row per model, one column per parameter
# ---------------------------------------------------------------------------
max_params = max(len(r["params"]) for r in all_results)
n_rows     = len(all_results)

fig = plt.figure(figsize=(max(3 * max_params, 14), 4 * n_rows))
gs  = gridspec.GridSpec(n_rows, max_params, figure=fig,
                        hspace=0.55, wspace=0.35)

color_chain = "#1f77b4"
color_shard = "#d62728"

for row, res in enumerate(all_results):
    for col, param in enumerate(res["params"]):
        ax = fig.add_subplot(gs[row, col])
        c  = res["chain"][param]
        s  = res["shard"][param]

        lo = min(c.min(), s.min()) - abs(c.std()) * 0.5
        hi = max(c.max(), s.max()) + abs(c.std()) * 0.5
        x  = np.linspace(lo, hi, 400)

        ax.fill_between(x, gaussian_kde(c)(x), alpha=0.35,
                        color=color_chain, label="chain×4")
        ax.fill_between(x, gaussian_kde(s)(x), alpha=0.35,
                        color=color_shard, label="shard×4")
        ax.plot(x, gaussian_kde(c)(x), color=color_chain, lw=1.5)
        ax.plot(x, gaussian_kde(s)(x), color=color_shard, lw=1.5,
                linestyle="--")

        kl = res["kls"][param]
        ax.set_title(f"{param}\nKL = {kl:.5f} nats", fontsize=8)
        ax.set_yticks([])
        ax.tick_params(labelsize=7)

        if row == 0 and col == 0:
            ax.legend(fontsize=7, loc="upper right")

    # Row label on the left
    fig.text(0.01, 1 - (row + 0.5) / n_rows,
             res["label"], va="center", ha="left",
             fontsize=9, fontweight="bold", rotation=90)

# Summary table below plot
summary_lines = []
for res in all_results:
    kl_vals = list(res["kls"].values())
    summary_lines.append(
        f"{res['label']:20s}  "
        f"mean KL={np.mean(kl_vals):.5f}  "
        f"max KL={np.max(kl_vals):.5f}  "
        f"params: {', '.join(f'{p}={v:.5f}' for p, v in res['kls'].items())}"
    )

fig.text(0.5, 0.01, "\n".join(summary_lines),
         ha="center", va="bottom", fontsize=7.5,
         fontfamily="monospace",
         bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.8))

plt.suptitle(
    "Posterior agreement: chain×4 vs shard×4\n"
    f"(blue=chain, red=shard  |  {NUM_CHAINS} chains × {N_SAMPLES} samples = "
    f"{NUM_CHAINS*N_SAMPLES} total  |  symmetric KL divergence)",
    fontsize=11, y=1.01,
)

plt.savefig("test_posterior_agreement.png", dpi=130, bbox_inches="tight")
print("\nPlot saved to test_posterior_agreement.png")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print(f"\n{'='*55}")
print(f"{'Model':<22} {'Param':<12} {'KL (nats)':>12}  {'status'}")
print("-" * 55)
for res in all_results:
    for p, kl in res["kls"].items():
        status = "✓ equivalent" if kl < 0.01 else ("~ close" if kl < 0.05 else "✗ differ")
        print(f"{res['label']:<22} {p:<12} {kl:>12.6f}  {status}")
print("="*55)
