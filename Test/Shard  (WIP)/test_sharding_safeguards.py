"""Exercise the sharding safeguards against every failure family.

Setup: 2 virtual CPU devices, N = 6 observations, mesh axis 'device'.
For each case we report what each safeguard detected and assert the expected
verdict. The cases are grouped so the OUTPUT itself documents the behaviour:

  SAFE              map / reduce               → no array collective, value matches
  COLLECTIVE        families 1,2,4,5           → static flags an array collective,
                                                 value still MATCHES (GSPMD keeps
                                                 results correct; risk is deadlock
                                                 /cost, which static catches)
  REDUCTION-ONLY    contraction over sharded   → only all-reduce (benign), safe
  SILENT            shard_map without psum      → static MISSES (no collective),
                                                 value check CATCHES the mismatch

Run:  python3 test_sharding_safeguards.py
"""
import os
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                           + " --xla_force_host_platform_device_count=2").strip()

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from jax import shard_map

# Import the SHIPPED module, not the stale 237-line fork that used to sit
# beside this file (it predated backward_collective_check entirely, so this
# suite was exercising code that is not what BayesForge ships).
import BayesForge.Diagnostic.sharding_safeguards as sg

jax.config.update("jax_enable_x64", True)

N_DEV = jax.local_device_count()
assert N_DEV == 2, f"expected 2 devices, got {N_DEV}"
MESH = Mesh(mesh_utils.create_device_mesh((N_DEV,)), axis_names=("device",))
DATA = NamedSharding(MESH, P("device"))
REP = NamedSharding(MESH, P())

N = 6
rng = np.random.default_rng(0)
y = rng.normal(size=N)
x = rng.normal(size=N)
Xmat = rng.normal(size=(N, N))
W = rng.normal(size=(3, N))
# symmetric positive-definite matrix for cholesky
S = Xmat @ Xmat.T + N * np.eye(N)
arange = jnp.arange(N, dtype=jnp.float64)


# ===========================================================================
# Cases:  (name, family, fn, args, shard_mask, expect_static, expect_runtime,
#          reference_fn)
#   expect_static  : "flag" | "safe"   (flagged by HLO collective OR primitive scan)
#   expect_runtime : "match" | "mismatch" | "error"
# ===========================================================================
def safe_map(a):
    return jnp.sum(a * 2.0 + 1.0)


def safe_reduce(a):
    return jax.scipy.special.logsumexp(a) + jnp.mean(a)


def fam1_lag(a):
    return jnp.sum(a[1:] - 0.5 * a[:-1])


def fam1_cumsum(a):
    return jnp.sum(jnp.cumsum(a) * arange)


def fam1_sort(a):
    return jnp.sum(jnp.sort(a) * arange)        # order-dependent reduction


def fam1_conv(a):
    k = jnp.array([0.25, 0.5, 0.25])
    return jnp.sum(jnp.convolve(a, k, mode="same") * arange)


def fam1_scan_over_axis(a):
    final, _ = lax.scan(lambda c, yt: (0.9 * c + yt, None), 0.0, a)  # scans axis 0
    return final


def fam2_transpose(M):
    return jnp.sum((M + M.T) * arange[:, None])


def fam2_outer(a):
    return jnp.sum(jnp.outer(a, a) * arange[:, None])


def fam3_contraction(a):                          # W (3,N) @ a(N,) contracts N
    return jnp.sum(W @ a)


def fam4_cholesky(M):
    return jnp.sum(jnp.linalg.cholesky(M))


def fam5_segment(a):
    # segment g1 spans the device boundary (rows 2 & 3)
    seg = jnp.array([0, 0, 1, 1, 2, 2])
    return jnp.sum(jax.ops.segment_sum(a, seg, num_segments=3) * jnp.arange(3.0))


def fam5_perm_gather(a):
    perm = jnp.array([4, 2, 5, 0, 3, 1])
    return jnp.sum(a[perm] * arange)


# --- SILENT family: shard_map result used incorrectly ---------------------
def fam6_shardmap_buggy(a):
    # The shard_map itself is valid (per-shard partial sums, out_specs match),
    # so it does NOT raise. The BUG is downstream: we keep only device 0's
    # partial and drop the rest — a *missing* reduction. No array collective is
    # inserted, so the static HLO scan sees nothing; only the value check, which
    # compares against the trusted total, catches that the result is wrong.
    f = shard_map(lambda xs: jnp.sum(xs, keepdims=True),     # (1,) per shard
                  mesh=MESH, in_specs=P("device"), out_specs=P("device"))
    partials = f(a)            # (n_dev,) — per-device partial sums
    return partials[0]         # BUG: only device 0's partial → silently wrong


def fam6_reference(a):     # trusted, sharding-free implementation of the intent
    return jnp.sum(a)


CASES = [
    ("safe_map",        "safe",   safe_map,             [x],    [True],  "safe",  "match",    None),
    ("safe_reduce",     "safe",   safe_reduce,          [x],    [True],  "safe",  "match",    None),
    ("fam1_lag",        "fam1",   fam1_lag,             [y],    [True],  "flag",  "match",    None),
    ("fam1_cumsum",     "fam1",   fam1_cumsum,          [y],    [True],  "flag",  "match",    None),
    ("fam1_sort",       "fam1",   fam1_sort,            [y],    [True],  "flag",  "match",    None),
    ("fam1_conv",       "fam1",   fam1_conv,            [y],    [True],  "flag",  "match",    None),
    ("fam1_scan",       "fam1",   fam1_scan_over_axis,  [y],    [True],  "flag",  "match",    None),
    ("fam2_transpose",  "fam2",   fam2_transpose,       [Xmat], [True],  "flag",  "match",    None),
    ("fam2_outer",      "fam2",   fam2_outer,           [x],    [True],  "flag",  "match",    None),
    ("fam3_contraction","fam3",   fam3_contraction,     [x],    [True],  "safe",  "match",    None),
    ("fam4_cholesky",   "fam4",   fam4_cholesky,        [S],    [True],  "flag",  "match",    None),
    ("fam5_segment",    "fam5",   fam5_segment,         [y],    [True],  "flag",  "match",    None),
    ("fam5_perm_gather","fam5",   fam5_perm_gather,     [y],    [True],  "flag",  "match",    None),
    ("fam6_shardmap",   "fam6",   fam6_shardmap_buggy,  [x],    [True],  "safe",  "mismatch", fam6_reference),
]


def fmt_static(rep):
    h = rep["hlo"]
    if h["lowering_error"]:
        return f"lowering-error ({h['lowering_error'][:30]}…)"
    cols = ", ".join(f"{k}×{v}" for k, v in h["array_collectives"].items())
    prim = rep["primitives"]["suspicious"]
    parts = []
    if cols:
        parts.append(cols)
    if prim:
        parts.append("prim:" + "/".join(prim))
    if not parts:
        return f"none (all-reduce×{h['reductions']})"
    return f"{'; '.join(parts)}  (all-reduce×{h['reductions']})"


def fmt_runtime(rep):
    v = rep["value"]
    if v["sharded_error"]:
        return f"ERROR: {v['sharded_error'][:36]}…"
    if v["match"]:
        return f"match (rep={v['replicated']:.3f}, shard={v['sharded']:.3f})"
    return f"MISMATCH (rep={v['replicated']:.3f}, shard={v['sharded']:.3f})"


def main():
    print(f"\ndevices={N_DEV}  N={N}  mesh axis='device'\n")
    header = f"{'case':<18}{'fam':<6}{'static (1)':<46}{'runtime (2)':<40}{'verdict'}"
    print(header)
    print("-" * len(header))

    passed = failed = 0
    for (name, fam, fn, args, mask, exp_static, exp_runtime, ref) in CASES:
        rep = sg.shard_safety_report(fn, args, mask, DATA, REP, reference_fn=ref)

        # ----- evaluate expectations -----
        # combined static = HLO collective scan (precise) OR primitive scan
        # (conservative; flags gather/scatter even when GSPMD finds a cheap path)
        got_static = "flag" if rep["static_unsafe"] else "safe"
        v = rep["value"]
        if v["sharded_error"]:
            got_runtime = "error"
        elif v["match"]:
            got_runtime = "match"
        else:
            got_runtime = "mismatch"

        ok_static = (got_static == exp_static)
        ok_runtime = (got_runtime == exp_runtime)
        ok = ok_static and ok_runtime
        passed += ok
        failed += (not ok)

        flag = "✓" if ok else "✗ FAIL"
        print(f"{name:<18}{fam:<6}{fmt_static(rep):<46}{fmt_runtime(rep):<40}{flag}")
        if not ok:
            print(f"    expected static={exp_static}/runtime={exp_runtime}, "
                  f"got static={got_static}/runtime={got_runtime}")

    print("-" * len(header))
    print(f"\n{passed} passed, {failed} failed\n")

    print("Reading the table:")
    print("  • SAFE map/reduce → no array collective, value matches.")
    print("  • Families 1,2,4,5 → static flags an ARRAY COLLECTIVE (deadlock/cost"
          " risk); value still MATCHES because GSPMD keeps results correct.")
    print("  • Contraction over the sharded axis → only a benign all-reduce → safe.")
    print("  • shard_map without psum → static MISSES it (a *missing* collective"
          " is invisible); the RUNTIME VALUE CHECK catches the mismatch.")
    print("  → the two safeguards are complementary; neither alone is sufficient.")

    return failed


if __name__ == "__main__":
    raise SystemExit(main())
