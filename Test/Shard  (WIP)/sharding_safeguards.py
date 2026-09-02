"""Two automatic safeguards for axis-0 data sharding.

1. STATIC check (run once at fit setup)
   ------------------------------------
   a) HLO-collective scan — lower+compile the function under the intended
      shardings and inspect the *optimized* HLO for array-moving collectives
      (all-gather / all-to-all / collective-permute / reduce-scatter). These are
      exactly what XLA inserts for the coupled failure families (cross-position,
      pairwise/transpose, linear algebra, cross-partition gather/scatter) and are
      the real danger: communication inside the gradient → deadlock under
      vectorized chains + lost speed-up. A scalar `all-reduce` (the likelihood
      sum / a pure reduction) is benign and is reported separately, not flagged.
   b) Jaxpr primitive scan — a coarse complement that flags the presence of
      deny-list primitives (cumsum, sort, cholesky, gather, ...).

2. RUNTIME value check
   -------------------
   Evaluate the function with the data REPLICATED (trusted reference) and with
   the data SHARDED, then compare. Under pure GSPMD the two agree by construction
   — so a *mismatch* means the model breaks sharding invariance (e.g. manual
   `shard_map` without the right collective, or device-dependent logic). Also
   catches a sharded path that raises while the replicated path does not.

These are complementary: the static scan catches the loud-but-correct collective
classes; the value check catches the silent numerical class the static scan
cannot see (a *missing* collective leaves no trace in the HLO).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


# Array-moving collectives — the red flags (deadlock / cost).
ARRAY_COLLECTIVES = ("all-gather", "all-to-all", "collective-permute",
                     "reduce-scatter")
# Reduction collective — benign (this is how a sum over the sharded axis,
# i.e. the log-likelihood, is aggregated).
REDUCTION_COLLECTIVE = "all-reduce"

# Coarse deny-list of jaxpr primitives whose presence is worth surfacing.
DENY_PRIMITIVES = {
    "cumsum", "cumprod", "cumlogsumexp", "cummax", "cummin",   # prefix scans
    "sort", "rev", "reduce_window",                            # reorder / window
    "conv_general_dilated", "fft",                             # conv / spectral
    "cholesky", "lu", "triangular_solve", "svd", "eig", "eigh",
    "geqrf", "householder_product",                           # linear algebra
    "gather", "scatter", "scatter-add", "scatter_add",        # index / scatter
}


# ---------------------------------------------------------------------------
# 1a. STATIC — HLO collective scan
# ---------------------------------------------------------------------------
def static_hlo_check(fn, sharded_args):
    """Compile *fn* under the committed input shardings and scan optimized HLO.

    Returns a dict:
        array_collectives : {opcode: count}  — flagged (array movement)
        reductions        : int              — all-reduce count (benign)
        lowering_error    : str | None       — set if lower/compile raised
        unsafe            : bool              — array_collectives or lowering error
    """
    try:
        compiled = jax.jit(fn).lower(*sharded_args).compile()
        hlo = compiled.as_text()
    except Exception as e:  # GSPMD cannot lower this sharding (itself a signal)
        return {"array_collectives": {}, "reductions": 0,
                "lowering_error": f"{type(e).__name__}: {e}", "unsafe": True}

    array_collectives = {op: hlo.count(op) for op in ARRAY_COLLECTIVES
                         if hlo.count(op) > 0}
    reductions = hlo.count(REDUCTION_COLLECTIVE)
    return {"array_collectives": array_collectives, "reductions": reductions,
            "lowering_error": None, "unsafe": bool(array_collectives)}


# ---------------------------------------------------------------------------
# 1b. STATIC — jaxpr primitive scan
# ---------------------------------------------------------------------------
def _collect_primitives(jaxpr, names):
    for eqn in jaxpr.eqns:
        names.add(eqn.primitive.name)
        for v in eqn.params.values():
            if hasattr(v, "eqns"):                       # nested jaxpr
                _collect_primitives(v, names)
            elif hasattr(v, "jaxpr") and hasattr(v.jaxpr, "eqns"):
                _collect_primitives(v.jaxpr, names)


def static_primitive_check(fn, args):
    """Flag deny-list primitives present anywhere in the traced jaxpr."""
    try:
        closed = jax.make_jaxpr(fn)(*args)
    except Exception as e:
        return {"suspicious": [], "trace_error": f"{type(e).__name__}: {e}"}
    names = set()
    _collect_primitives(closed.jaxpr, names)
    return {"suspicious": sorted(names & DENY_PRIMITIVES), "trace_error": None}


# ---------------------------------------------------------------------------
# 2. RUNTIME value check
# ---------------------------------------------------------------------------
def runtime_value_check(fn, args, arg_shardings, rep_sharding,
                        reference_fn=None, tol=1e-5):
    """Compare a REPLICATED reference run against the SHARDED run.

    reference_fn defaults to *fn* run with replicated inputs (correct for plain
    GSPMD models). Pass an independent trusted implementation when *fn* itself
    embeds sharding (e.g. shard_map), so the reference is not also broken.

    Returns a dict:
        replicated, sharded : float | None  (scalar summaries; None on error)
        match               : bool | None
        sharded_error       : str | None    (sharded path raised)
        unsafe              : bool
    """
    ref_fn = reference_fn or fn
    rep_args = [jax.device_put(np.asarray(a), rep_sharding) for a in args]
    val_ref = np.asarray(ref_fn(*rep_args))

    try:
        sh_args = [jax.device_put(np.asarray(a), s)
                   for a, s in zip(args, arg_shardings)]
        out = fn(*sh_args)
        out.block_until_ready()
        val_sh = np.asarray(out)
        match = bool(np.allclose(val_ref, val_sh, rtol=tol, atol=tol))
        return {"replicated": float(np.sum(val_ref)),
                "sharded": float(np.sum(val_sh)),
                "match": match, "sharded_error": None, "unsafe": not match}
    except Exception as e:
        return {"replicated": float(np.sum(val_ref)), "sharded": None,
                "match": None, "sharded_error": f"{type(e).__name__}: {e}",
                "unsafe": True}


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------
def shard_safety_report(fn, args, shard_mask, data_sharding, rep_sharding,
                        reference_fn=None, tol=1e-5):
    """Run all three checks. *shard_mask* is a list of bools, one per arg."""
    arg_shardings = [data_sharding if m else rep_sharding for m in shard_mask]
    sharded_args = [jax.device_put(np.asarray(a), s)
                    for a, s in zip(args, arg_shardings)]

    hlo = static_hlo_check(fn, sharded_args)
    prim = static_primitive_check(fn, args)
    val = runtime_value_check(fn, args, arg_shardings, rep_sharding,
                              reference_fn=reference_fn, tol=tol)
    return {"hlo": hlo, "primitives": prim, "value": val,
            "static_unsafe": hlo["unsafe"] or bool(prim["suspicious"]),
            "runtime_unsafe": val["unsafe"]}


# ---------------------------------------------------------------------------
# Human-readable diagnosis — turn a report into severity-tagged messages
# ---------------------------------------------------------------------------
_LINALG = {"cholesky", "lu", "triangular_solve", "svd", "eig", "eigh",
           "geqrf", "householder_product"}
_INDEX = {"gather", "scatter", "scatter-add", "scatter_add"}
_CROSSPOS = {"cumsum", "cumprod", "cumlogsumexp", "cummax", "cummin",
             "sort", "rev", "reduce_window", "conv_general_dilated", "fft"}

_COLLECTIVE_MSG = {
    "all-to-all": ("⚠", "transpose / pairwise pattern (e.g. Y.T, outer product, "
                   "attention) — the most expensive collective and the highest "
                   "deadlock risk under vectorized chains"),
    "all-gather": ("⚠", "a full axis is gathered onto every device — NO memory "
                   "relief (each device holds the whole array); the model likely "
                   "uses cumsum / sort / scan / cholesky over the sharded axis"),
    "collective-permute": ("⚠", "neighbour / halo exchange (lag, diff, "
                           "convolution) — correct, but a per-step sync point → "
                           "deadlock risk under vectorized chains"),
    "reduce-scatter": ("⚠", "a contraction communicates partial sums across "
                       "devices — usually cheap, but still a sync point"),
}


def diagnose(report):
    """Return a list of (severity, message) tuples summarising *report*.

    severity ∈ {'✓','ℹ','⚠','✗'}. Intended to be printed only when sharding is
    enabled (shard=True). Correctness verdicts come first.
    """
    msgs = []
    v, hlo, prim = report["value"], report["hlo"], report["primitives"]

    # --- correctness (✗) ---------------------------------------------------
    if hlo["lowering_error"]:
        msgs.append(("✗", f"this sharding cannot be compiled "
                     f"({hlo['lowering_error']}); replicate the offending array"))
    if v["sharded_error"]:
        msgs.append(("✗", f"sharded execution raised: {v['sharded_error']}"))
    elif v["match"] is False:
        delta = abs((v["replicated"] or 0.0) - (v["sharded"] or 0.0))
        msgs.append(("✗", f"sharded result differs from the replicated reference "
                     f"by {delta:.3g} — sharding CHANGES the answer (likely a "
                     f"shard_map without psum); do NOT shard this model"))

    # --- cost / liveness (⚠) ----------------------------------------------
    for op, n in hlo["array_collectives"].items():
        sev, text = _COLLECTIVE_MSG.get(op, ("⚠", f"array collective {op}"))
        msgs.append((sev, f"{text} [{op}×{n}]"))

    susp = set(prim["suspicious"])
    if susp & _LINALG:
        msgs.append(("⚠", f"whole-matrix linear algebra ({', '.join(sorted(susp & _LINALG))}) "
                     "on a sharded array — replicate it, or exploit block / "
                     "low-rank / Kronecker structure"))
    if susp & _CROSSPOS:
        msgs.append(("⚠", f"cross-position op(s) ({', '.join(sorted(susp & _CROSSPOS))}) "
                     "over the sharded axis — couples rows across the boundary"))
    if (susp & _INDEX) and not hlo["array_collectives"]:
        msgs.append(("ℹ", f"index / segment op ({', '.join(sorted(susp & _INDEX))}) — "
                     "GSPMD found a local path here, but correctness across "
                     "partitions depends on the index VALUES; verify they do not "
                     "cross the device boundary"))

    # --- all clear (✓) -----------------------------------------------------
    if not msgs:
        msgs.append(("✓", f"sharding compatible — map + reduce over axis 0, no "
                     f"array collectives (all-reduce×{hlo['reductions']} only); "
                     "clean parallelism expected"))
    return msgs


def print_diagnosis(report, name=""):
    """Pretty-print :func:`diagnose` output."""
    head = f"shard check{' — ' + name if name else ''}:"
    print(head)
    for sev, text in diagnose(report):
        print(f"  {sev} {text}")
