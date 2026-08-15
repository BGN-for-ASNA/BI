"""Test manual per-operation sharding declared INSIDE a model body.

Each "model" is a function f(*data) -> scalar (a stand-in for a log-density).
The data is kept REPLICATED; only a chosen operation is sharded, declared inside
the model with the helpers in manual_shard.py. For every model we check:

  correctness : sharded result == trusted replicated reference  (value check)
  cost        : which collectives XLA inserted                  (static HLO scan)

Setup: 2 virtual CPU devices, N = 6.  Run:  python3 test_manual_ops.py
"""
import os
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                           + " --xla_force_host_platform_device_count=2").strip()
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

import manual_shard as ms
import sharding_safeguards as sg          # reused from the parent folder

jax.config.update("jax_enable_x64", True)

N_DEV = jax.local_device_count()
assert N_DEV == 2, f"expected 2 devices, got {N_DEV}"
MESH = ms.make_mesh()
from jax.sharding import NamedSharding, PartitionSpec as P
DATA = NamedSharding(MESH, P("device"))
REP = NamedSharding(MESH, P())

N = 6
rng = np.random.default_rng(0)
X = rng.normal(size=(N, 3))
s_vec = rng.normal(size=N)
r_vec = rng.normal(size=N)


def heavy_rowwise(Xrow):
    """Expensive per-row map we want to parallelize (cheap here, stands in)."""
    return jnp.tanh(Xrow @ jnp.array([0.5, -0.3, 0.2])) + 1.0


# ===========================================================================
# Models with manual per-op sharding
# ===========================================================================

# --- M1: GSPMD constraint — shard ONE op, keep data replicated -------------
def m1_constraint_shard_one_op(Xrep):
    Xs = ms.shard0(Xrep, MESH)             # declare: run the next op sharded
    feats = jax.vmap(heavy_rowwise)(Xs)    # heavy per-row op runs partitioned
    return jnp.sum(feats)                  # reduce


def m1_reference(Xrep):
    return jnp.sum(jax.vmap(heavy_rowwise)(Xrep))


# --- M2: explicit shard_map + psum (no hidden comms) -----------------------
def m2_shardmap_mapreduce(Xrep):
    f = ms.map_reduce_sum(lambda Xsh: jnp.sum(jax.vmap(heavy_rowwise)(Xsh)),
                          MESH, n_sharded=1)
    return f(Xrep)


# --- M3: SRM outer sum — shard s, replicate r, no transpose ----------------
def m3_srm_outer(s, r):
    M = ms.outer_sum_srm(s, r, MESH)       # (N, N) sharded on axis 0
    return jnp.sum(M * jnp.arange(N, dtype=jnp.float64)[:, None])


def m3_reference(s, r):
    M = s[:, None] + r[None, :]
    return jnp.sum(M * jnp.arange(N, dtype=jnp.float64)[:, None])


# --- M4: DELIBERATELY WRONG shard_map (missing psum) -----------------------
def m4_shardmap_buggy(Xrep):
    f = ms.map_reduce_sum_BUGGY(lambda Xsh: jnp.sum(jax.vmap(heavy_rowwise)(Xsh)),
                                MESH, n_sharded=1)
    return f(Xrep)


CASES = [
    # name, model, args, shard_mask, reference, expect_correct
    ("M1 constraint: shard one op", m1_constraint_shard_one_op, [X],
     [True], m1_reference, True),
    ("M2 shard_map map+reduce",     m2_shardmap_mapreduce,       [X],
     [True], m1_reference, True),
    ("M3 SRM outer sum (s shard)",  m3_srm_outer,                [s_vec, r_vec],
     [True, False], m3_reference, True),
    ("M4 shard_map MISSING psum",   m4_shardmap_buggy,           [X],
     [True], m1_reference, False),   # expected WRONG → value check must catch
]


def fmt_static(rep):
    h = rep["hlo"]
    if h["lowering_error"]:
        return f"lowering-error"
    cols = ", ".join(f"{k}×{v}" for k, v in h["array_collectives"].items())
    return cols if cols else f"none (all-reduce×{h['reductions']})"


def main():
    print(f"\ndevices={N_DEV}  N={N}  — manual per-op sharding inside the model\n")
    header = f"{'model':<32}{'correct?':<22}{'collectives (cost)':<30}{'verdict'}"
    print(header); print("-" * len(header))

    passed = failed = 0
    for name, model, args, mask, ref, expect_correct in CASES:
        rep = sg.shard_safety_report(model, args, mask, DATA, REP, reference_fn=ref)
        v = rep["value"]
        if v["sharded_error"]:
            correct = False
            corr_str = f"ERROR: {v['sharded_error'][:18]}…"
        else:
            correct = bool(v["match"])
            corr_str = (f"match ({v['sharded']:.3f})" if correct
                        else f"WRONG (ref={v['replicated']:.3f}, got={v['sharded']:.3f})")

        ok = (correct == expect_correct)
        passed += ok; failed += (not ok)
        flag = "✓" if ok else "✗ FAIL"
        print(f"{name:<32}{corr_str:<22}{fmt_static(rep):<30}{flag}")

    print("-" * len(header))
    print(f"\n{passed} passed, {failed} failed\n")
    print("Reading the table:")
    print("  • M1 (with_sharding_constraint): correct — GSPMD guarantees it. The")
    print("    constraint is a layout HINT; here (small N) XLA optimized it away,")
    print("    so no array collective appears. At scale it inserts a scatter to")
    print("    shard the op and a gather to reassemble — that is its cost.")
    print("  • M2 (shard_map + psum): correct AND no array collective — the")
    print("    explicit map+reduce, communication only the scalar psum.")
    print("  • M3 (SRM outer sum): correct, no transpose/all-to-all — shard s,")
    print("    replicate r, local outer sum.")
    print("  • M4 (shard_map WITHOUT psum): silently WRONG — manual sharding puts")
    print("    correctness on the author; only the value check catches it.")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
