"""Verify the backward-pass collective-volume safeguard.

Two contrasting models, data sharded over a 'data' axis:

  favorable   : independent Bernoulli logits a + b*X  → grad all-reduces only the
                O(1) scalar params → communication-light → must NOT be flagged.
  coupled/SRM : a replicated O(N²) parameter gathered into a sharded (N,N)
                likelihood → grad all-reduces the O(N²) param every step →
                O(N²)-dominant → MUST be flagged (dominant=True).
"""
import os, sys
CORES = int(os.environ.get("CORES", "8"))
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={CORES}"

import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpyro, numpyro.distributions as ndist
from numpyro import sample
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, "/home/sebastian_sosa/BF")
from BayesForge.Diagnostic.sharding_safeguards import run_shard_check

NDEV = int(os.environ.get("NDEV", "4"))
N    = int(os.environ.get("N", "80"))
ND   = N * (N - 1) // 2
mesh = Mesh(np.array(jax.devices()[:NDEV]), ("data",))
key  = jax.random.PRNGKey(0)
print(f"devices={CORES} n_shards={NDEV} N={N} N_dyads={ND}\n")


def put(a, spec):
    return jax.device_put(jnp.asarray(a), NamedSharding(mesh, spec))


# ── favorable: independent likelihood ─────────────────────────────────────────
def fav_model(X, Y):
    a = sample("a", ndist.Normal(0, 10)); b = sample("b", ndist.Normal(0, 10))
    s = sample("s", ndist.Exponential(1.0))
    sample("Y", ndist.Normal(a + b * X, s), obs=Y)

M = N * N
Xf = np.random.normal(size=M); Yf = 2 + 0.8 * Xf + np.random.normal(size=M)
fav_shard = dict(X=put(Xf, P("data")), Y=put(Yf, P("data")))
fav_rep   = dict(X=put(Xf, P()),       Y=put(Yf, P()))

# ── coupled / SRM-like: replicated O(N²) param gathered into sharded matrix ────
idxmap = np.random.randint(0, ND, size=(N, N)).astype(np.int32)
Yc     = np.random.binomial(1, 0.3, size=(N, N)).astype(np.float64)

def coupled_model(Y, idx):
    dr = sample("dr", ndist.Normal(0, 1).expand([ND]).to_event(1))  # replicated O(N²)
    logits = dr[idx]                                                # (N,N) gather
    sample("Y", ndist.Bernoulli(logits=logits), obs=Y)

cpl_shard = dict(Y=put(Yc, P("data")),     idx=put(idxmap, P("data")))
cpl_rep   = dict(Y=put(Yc, P()),           idx=put(idxmap, P()))


def summarize(tag, report):
    b = report["backward"]
    print(f"\n[{tag}] perf_unsafe={report['perf_unsafe']}  "
          f"coll_elems={b['coll_elems']}  ratio={b['ratio']:.3f}  "
          f"dominant={b['dominant']}")


print("=" * 70, "\nFAVORABLE (independent likelihood) — expect dominant=False")
r1 = run_shard_check(fav_model, fav_shard, fav_rep, key, name="favorable")
summarize("favorable", r1)

print("\n" + "=" * 70, "\nCOUPLED / SRM-like — expect dominant=True")
r2 = run_shard_check(coupled_model, cpl_shard, cpl_rep, key, name="coupled")
summarize("coupled", r2)

ok = (r1["perf_unsafe"] is False) and (r2["perf_unsafe"] is True)
print("\n" + "=" * 70)
print("RESULT:", "PASS ✓" if ok else "FAIL ✗")
sys.exit(0 if ok else 1)
