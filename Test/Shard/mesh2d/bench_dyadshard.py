"""Does sharding by DYAD (not by row) make the SRM speed up?

User's insight: shard the EXPENSIVE O(N^2) dyadic random effect along the dyad
axis so its gradient stays LOCAL (no all-reduce), and pay communication only on
the CHEAP O(N) nodal effects — gathered redundantly into each shard's dyad block.

  row-sharded  (bench_bigN): dyadic param replicated → O(N^2) all-reduce  → loses
  dyad-sharded (this file)  : dyadic param local      → O(N)   all-reduce  → ?

Layout: everything in edgelist/dyad space. Each of the N_dyads unordered dyads d
carries endpoints (i_d, j_d), both directed outcomes (y_ij, y_ji) and predictors.
Per-dyad arrays are sharded on axis 0; dr_raw (2, N_dyads) is pinned sharded on
its dyad axis via with_sharding_constraint. Nodal s, r (N,) stay replicated and are
gathered: s[i_d], r[j_d], ... (forward local; gradient scatter-add = O(N) all-reduce).

Steady-state s/it, compile removed (build jitted loop once, compile via first call,
time a cached second call). Sweeps ndata to show the trend.

Run:  PYTHONPATH=/home/sebastian_sosa/BF CORES=20 N=400 python3 bench_dyadshard.py
"""
import os, sys, time
CORES = int(os.environ.get("CORES", "20"))
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={CORES}"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax, jax.numpy as jnp
from jax.lax import with_sharding_constraint
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpyro, numpyro.distributions as ndist
from numpyro import sample
from numpyro.infer import NUTS
from mesh2d_nuts import single_chain_loop
jax.config.update("jax_enable_x64", True)

N     = int(os.environ.get("N", "400"))
TREE  = int(os.environ.get("TREE", "5"))
ITERS = int(os.environ.get("ITERS", "20"))
SWEEP = [int(x) for x in os.environ.get("SWEEP", "1,5,10,20").split(",")]
SEED  = 42
K     = 3
ND    = N * (N - 1) // 2
print(f"N={N} N_dyads={ND} cores={CORES} tree_cap={TREE} iters={ITERS} sweep={SWEEP}\n")
sys.stdout.flush()

# ── simulate a network, lay it out in dyad space ──────────────────────────────
np.random.seed(SEED)
ur, uc = np.triu_indices(N, 1)                      # (ND,) endpoints per dyad
i_idx = jnp.asarray(ur, jnp.int32); j_idx = jnp.asarray(uc, jnp.int32)
s_t = np.random.normal(0, 1, N); r_t = np.random.normal(0, 1, N)
pred_ij = np.random.normal(0, 1, (ND, K)); pred_ji = np.random.normal(0, 1, (ND, K))
beta_t = np.array([0.5, -0.3, 0.2])
lo_ij = s_t[ur] + r_t[uc] + pred_ij @ beta_t + np.random.normal(0, 1, ND)
lo_ji = s_t[uc] + r_t[ur] + pred_ji @ beta_t + np.random.normal(0, 1, ND)
y_ij = jnp.asarray((np.random.rand(ND) < 1/(1+np.exp(-lo_ij))).astype(np.float64))
y_ji = jnp.asarray((np.random.rand(ND) < 1/(1+np.exp(-lo_ji))).astype(np.float64))
pred_ij = jnp.asarray(pred_ij); pred_ji = jnp.asarray(pred_ji)

MESH = None   # set per-config; the model reads it for the sharding constraint


def srm_dyad(i_idx, j_idx, y_ij, y_ji, pred_ij, pred_ji):
    # nodal random effects (replicated, O(N))
    sr_raw = sample("sr_raw", ndist.Normal(0, 1).expand([2, N]).to_event(2))
    sr_sig = sample("sr_sigma", ndist.TruncatedNormal(0, 2.5, low=0).expand([2]).to_event(1))
    sr_L   = sample("sr_L", ndist.LKJCholesky(2, 2.5))
    sr_rf  = (sr_sig[:, None] * sr_L) @ sr_raw      # (2, N)
    s, r   = sr_rf[0], sr_rf[1]                     # (N,) replicated
    b0     = sample("b0", ndist.Normal(0, 2.5))
    beta   = sample("beta", ndist.Normal(0, 2.5).expand([K]).to_event(1))

    # dyadic random effects (2, N_dyads) — PIN sharded on the dyad axis
    dr_raw = sample("dr_raw", ndist.Normal(0, 1).expand([2, ND]).to_event(2))
    if MESH is not None:
        dr_raw = with_sharding_constraint(dr_raw, NamedSharding(MESH, P(None, "data")))
    dr_sig = sample("dr_sigma", ndist.TruncatedNormal(0, 2.5, low=0))
    dr_L   = sample("dr_L", ndist.LKJCholesky(2, 2.5))
    dr     = (dr_sig * dr_L) @ dr_raw               # (2, N_dyads) sharded on dyad
    dr_ij, dr_ji = dr[0], dr[1]                     # (N_dyads,) sharded

    ff_ij = pred_ij @ beta; ff_ji = pred_ji @ beta  # (N_dyads,) sharded
    # gather nodal effects into the local dyad block (forward local; grad → O(N) all-reduce)
    logit_ij = b0 + s[i_idx] + r[j_idx] + ff_ij + dr_ij
    logit_ji = b0 + s[j_idx] + r[i_idx] + ff_ji + dr_ji
    sample("y_ij", ndist.Bernoulli(logits=logit_ij), obs=y_ij)
    sample("y_ji", ndist.Bernoulli(logits=logit_ji), obs=y_ji)


KW = dict(i_idx=i_idx, j_idx=j_idx, y_ij=y_ij, y_ji=y_ji,
          pred_ij=pred_ij, pred_ji=pred_ji)
DYAD_SPEC = {"i_idx": P("data"), "j_idx": P("data"), "y_ij": P("data"),
             "y_ji": P("data"), "pred_ij": P("data"), "pred_ji": P("data")}


def build_and_time(ndata, label):
    global MESH
    if ND % ndata != 0:
        print(f"{label:<22} SKIP (N_dyads={ND} not divisible by {ndata})"); return None
    devs = np.array(jax.devices()[:ndata]).reshape(ndata)
    MESH = Mesh(devs, ("data",))
    specs = DYAD_SPEC if ndata > 1 else {k: P() for k in KW}
    kernel = NUTS(srm_dyad, max_tree_depth=TREE, target_accept_prob=0.8)

    def loop(rk, mk):
        return single_chain_loop(kernel, rk, 1, ITERS, mk)

    mk = {k: jax.device_put(v, NamedSharding(MESH, specs.get(k, P())))
          for k, v in KW.items()}
    jfn = jax.jit(loop)
    key = jax.random.PRNGKey(1)

    out = jfn(key, mk); jax.block_until_ready([v for v in out.values()])
    t0 = time.perf_counter()
    out = jfn(key, mk); jax.block_until_ready([v for v in out.values()])
    t_exec = time.perf_counter() - t0
    rate = t_exec / ITERS
    print(f"{label:<22} exec={t_exec:7.2f}s  steady={rate:6.3f} s/it"); sys.stdout.flush()
    return rate


if __name__ == "__main__":
    print(f">>> DYAD-sharded SRM, single chain, sweep at N={N}\n")
    base = build_and_time(1, "ndata=1 (unsharded)")
    rows = [(1, base)]
    for nd in SWEEP:
        if nd == 1: continue
        r = build_and_time(nd, f"ndata={nd}")
        if r is not None: rows.append((nd, r))
    print("\n=== DYAD-SHARDED steady-state s/it (compile removed) ===")
    print(f"{'ndata':>6} {'s/it':>9} {'speedup vs 1':>14}")
    for nd, r in rows:
        sp = base / r if (r and base) else float("nan")
        print(f"{nd:>6} {r:>9.3f} {sp:>13.2f}x")
    if len(rows) > 1:
        best_nd, best_r = min(rows, key=lambda x: x[1])
        print(f"\nbest = ndata={best_nd} at {base/best_r:.2f}x vs unsharded")
