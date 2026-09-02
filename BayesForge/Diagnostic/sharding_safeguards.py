"""Automatic safeguards for axis-0 data sharding, integrated with numpyro.

Two checks, run ONCE at fit setup (never inside the MCMC loop):

1. STATIC — compile the model's log-density under the intended shardings and
   scan the optimized HLO for array-moving collectives (all-gather / all-to-all /
   collective-permute / reduce-scatter). These are what XLA inserts for the
   coupled operations and are the real risk: communication inside the gradient →
   deadlock under vectorized chains + lost speed-up. A scalar all-reduce (the
   log-likelihood sum) is benign and reported separately. A coarse jaxpr
   primitive scan complements it for index ops (gather/scatter/...).

2. RUNTIME value — evaluate the log-density with data REPLICATED vs SHARDED at
   the same parameters and compare. Under GSPMD these agree by construction, so a
   mismatch means the model breaks sharding invariance (e.g. manual shard_map
   without psum). This is the only check that catches a silently-wrong number.

``run_shard_check`` ties both to a numpyro model via ``initialize_model`` and is
best-effort: any failure warns and returns None rather than blocking the fit.
"""
from __future__ import annotations

import os
import re
import numpy as np
import jax
import jax.numpy as jnp

ARRAY_COLLECTIVES = ("all-gather", "all-to-all", "collective-permute",
                     "reduce-scatter")
REDUCTION_COLLECTIVE = "all-reduce"

DENY_PRIMITIVES = {
    "cumsum", "cumprod", "cumlogsumexp", "cummax", "cummin",
    "sort", "rev", "reduce_window", "conv_general_dilated", "fft",
    "cholesky", "lu", "triangular_solve", "svd", "eig", "eigh",
    "geqrf", "householder_product",
    "gather", "scatter", "scatter-add", "scatter_add",
}

_LINALG = {"cholesky", "lu", "triangular_solve", "svd", "eig", "eigh",
           "geqrf", "householder_product"}
_INDEX = {"gather", "scatter", "scatter-add", "scatter_add"}
_CROSSPOS = {"cumsum", "cumprod", "cumlogsumexp", "cummax", "cummin",
             "sort", "rev", "reduce_window", "conv_general_dilated", "fft"}

_COLLECTIVE_MSG = {
    "all-to-all": ("⚠", "transpose / pairwise pattern (e.g. Y.T, outer product) — "
                   "the most expensive collective and the highest deadlock risk "
                   "under vectorized chains"),
    "all-gather": ("⚠", "a full axis is gathered onto every device — NO memory "
                   "relief; the model likely uses cumsum / sort / scan / cholesky "
                   "over the sharded axis"),
    "collective-permute": ("⚠", "neighbour / halo exchange (lag, diff, conv) — "
                           "correct, but a per-step sync point → deadlock risk"),
    "reduce-scatter": ("⚠", "a contraction communicates partial sums across "
                       "devices — usually cheap, but still a sync point"),
}


# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------
def _count_collective(hlo: str, op: str) -> int:
    """Count HLO *instructions* of a collective opcode.

    A plain ``hlo.count(op)`` double-counts async collectives (XLA emits
    ``all-gather-start`` and ``all-gather-done`` as separate instructions, both
    containing the opcode) and also matches the opcode inside ``op_name``
    metadata strings. Match the operator position instead: ``%x = shape op(``.
    """
    pattern = re.compile(
        rf"=\s*\S+\s+{re.escape(op)}(?:-start|-done)?\s*\(", re.MULTILINE)
    n = 0
    for line in hlo.splitlines():
        if pattern.search(line):
            # -done pairs with its -start; count the pair once.
            if f"{op}-done" in line:
                continue
            n += 1
    return n


def static_hlo_check(fn, args):
    try:
        hlo = jax.jit(fn).lower(*args).compile().as_text()
    except Exception as e:
        return {"array_collectives": {}, "reductions": 0,
                "lowering_error": f"{type(e).__name__}: {e}", "unsafe": True}
    array_collectives = {}
    for op in ARRAY_COLLECTIVES:
        n = _count_collective(hlo, op)
        if n > 0:
            array_collectives[op] = n
    return {"array_collectives": array_collectives,
            "reductions": _count_collective(hlo, REDUCTION_COLLECTIVE),
            "lowering_error": None, "unsafe": bool(array_collectives)}


def _collect_primitives(jaxpr, names, _seen=None):
    if _seen is None:
        _seen = set()
    if id(jaxpr) in _seen:
        return
    _seen.add(id(jaxpr))
    for eqn in jaxpr.eqns:
        names.add(eqn.primitive.name)
        for v in eqn.params.values():
            # Params can hold a Jaxpr, a ClosedJaxpr, or a tuple/list of them
            # (lax.cond stores `branches` as a tuple). Only recursing into bare
            # values missed every primitive inside a cond branch.
            for sub in (v if isinstance(v, (list, tuple)) else (v,)):
                if hasattr(sub, "eqns"):
                    _collect_primitives(sub, names, _seen)
                elif hasattr(sub, "jaxpr") and hasattr(sub.jaxpr, "eqns"):
                    _collect_primitives(sub.jaxpr, names, _seen)


def static_primitive_check(fn, args):
    try:
        closed = jax.make_jaxpr(fn)(*args)
    except Exception as e:
        return {"suspicious": [], "trace_error": f"{type(e).__name__}: {e}"}
    names = set()
    _collect_primitives(closed.jaxpr, names)
    return {"suspicious": sorted(names & DENY_PRIMITIVES), "trace_error": None}


# ---------------------------------------------------------------------------
# Backward-pass collective volume — the check that catches the SRM pathology
# ---------------------------------------------------------------------------
# The forward log-density of a correctly-sharded model is often collective-free
# (gathers/outer-sums are GSPMD-local); the expensive collective lives in the
# GRADIENT, where the per-shard contributions to a REPLICATED parameter must be
# all-reduced. That all-reduce does not shrink with the device count, so when its
# volume is the same order as the per-shard compute it parallelises (O(N²) vs
# O(N²)), sharding cannot win however many devices are added. We estimate it by
# compiling grad(potential) under the intended shardings and measuring the
# collective byte-volume relative to the per-shard data it is meant to accelerate.

_ALL_COLLECTIVES = ("all-reduce", "all-gather", "all-to-all",
                    "reduce-scatter", "collective-permute")
# HLO shape tokens, e.g. f64[79800], s32[400,400], bf16[]  → capture the dims
_SHAPE_RE = re.compile(r"(?:f|s|u|bf|c|pred)\d*\[([0-9,]*)\]")


def _shape_elems(dims_str):
    dims = [int(d) for d in dims_str.split(",") if d != ""]
    n = 1
    for d in dims:
        n *= d
    return n  # scalar "[]" → 1


def _collective_volume_from_hlo(hlo):
    """Total element-count moved by collectives, per leapfrog (estimate)."""
    total = 0
    largest = 0
    for line in hlo.splitlines():
        if any(op in line for op in _ALL_COLLECTIVES):
            elems = [_shape_elems(m.group(1)) for m in _SHAPE_RE.finditer(line)]
            if elems:
                line_max = max(elems)          # the array this collective moves
                total += line_max
                largest = max(largest, line_max)
    return total, largest


def _infer_n_shards(kwargs):
    """Largest partition factor across the (committed) data arrays; 1 if none."""
    n = 1
    for v in kwargs.values():
        try:
            shard = v.sharding.shard_shape(v.shape)
            factor = int(round(np.prod(v.shape) / max(np.prod(shard), 1)))
            n = max(n, factor)
        except Exception:
            pass
    return n


def _sharded_data_elems(kwargs):
    """Total elements of arrays that are actually partitioned (the parallel work).

    Returns 0 when nothing is partitioned; falling back to the total size of
    every array made `ratio` look small for a reason unrelated to sharding.
    """
    sharded = 0
    for v in kwargs.values():
        sz = int(np.prod(getattr(v, "shape", ()) or (1,)))
        try:
            shard = v.sharding.shard_shape(v.shape)
            if int(np.prod(shard)) < sz:
                sharded += sz
        except Exception:
            pass
    return sharded


def backward_collective_check(model, params, sharded_kwargs,
                              ratio_threshold=None):
    """Compile grad(potential) under the intended shardings and decide whether
    inter-device communication is O(N²)-dominant (comparable to the per-shard
    compute it parallelises). Returns a report; never raises.

    The data is passed as an EXPLICIT argument (with its committed sharding) so
    GSPMD propagates and the gradient's parameter all-reduce actually appears in
    the HLO — lowering with the data merely closed over would compile single-device
    and hide every collective.

    dominant == True  → sharding cannot pay off; caller should replicate.
    """
    if ratio_threshold is None:
        ratio_threshold = float(os.environ.get("BF_SHARD_PERF_RATIO", "0.25"))
    try:
        from numpyro.infer.util import potential_energy

        def _pot(p, kw):
            return potential_energy(model, (), kw, p)

        grad_fn = jax.grad(_pot, argnums=0)
        hlo = jax.jit(grad_fn).lower(params, sharded_kwargs).compile().as_text()
    except Exception as e:
        return {"coll_elems": None, "dominant": False, "ratio": None,
                "n_shards": None, "error": f"{type(e).__name__}: {e}"}
    coll_total, coll_largest = _collective_volume_from_hlo(hlo)
    n_shards = _infer_n_shards(sharded_kwargs)
    data_elems = _sharded_data_elems(sharded_kwargs)
    per_shard = data_elems / max(n_shards, 1)
    # comm volume relative to the per-shard compute it is meant to accelerate.
    ratio = (coll_total / per_shard) if per_shard > 0 else float("inf")
    dominant = (n_shards > 1) and (ratio >= ratio_threshold)
    return {"coll_elems": coll_total, "coll_largest": coll_largest,
            "n_shards": n_shards, "data_elems": data_elems,
            "ratio": ratio, "ratio_threshold": ratio_threshold,
            "dominant": dominant, "error": None}


# ---------------------------------------------------------------------------
# Runtime value check
# ---------------------------------------------------------------------------
def runtime_value_check(fn_sharded, fn_ref, params, tol=1e-4):
    # The reference call has to be inside a guard too, otherwise its failure is
    # swallowed by run_shard_check's blanket except and reported only as
    # "shard-check skipped", losing the actual error.
    try:
        val_ref = np.asarray(fn_ref(params))
    except Exception as e:
        return {"replicated": None, "sharded": None, "match": None,
                "replicated_error": f"{type(e).__name__}: {e}",
                "sharded_error": None, "unsafe": True}
    try:
        out = fn_sharded(params)
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        val_sh = np.asarray(out)
        match = bool(np.allclose(val_ref, val_sh, rtol=tol, atol=tol))
        return {"replicated": float(np.sum(val_ref)), "sharded": float(np.sum(val_sh)),
                "match": match, "replicated_error": None,
                "sharded_error": None, "unsafe": not match}
    except Exception as e:
        return {"replicated": float(np.sum(val_ref)), "sharded": None, "match": None,
                "replicated_error": None,
                "sharded_error": f"{type(e).__name__}: {e}", "unsafe": True}


def shard_safety_report(fn, args, shard_mask, data_sharding, rep_sharding,
                        reference_fn=None, tol=1e-5):
    """Static + runtime shard check on a raw JAX function.

    The function-level counterpart to :func:`run_shard_check`, which takes a
    numpyro model. ``shard_mask`` is a list of bools, one per positional arg:
    True places that argument on ``data_sharding``, False on ``rep_sharding``.

    Args:
        fn: the function to check.
        args: its positional arguments (host arrays).
        shard_mask: one bool per arg — shard it or replicate it.
        data_sharding: sharding for the masked-True args.
        rep_sharding: sharding for everything else, and for the reference run.
        reference_fn: trusted implementation to compare against; defaults to
            ``fn`` evaluated on fully replicated inputs.
        tol: rtol/atol for the value comparison.

    Returns:
        A report dict in the same shape ``diagnose`` and ``print_diagnosis``
        consume.
    """
    arg_shardings = [data_sharding if m else rep_sharding for m in shard_mask]
    sharded_args = [jax.device_put(np.asarray(a), s)
                    for a, s in zip(args, arg_shardings)]
    rep_args = [jax.device_put(np.asarray(a), rep_sharding) for a in args]

    ref_fn = reference_fn if reference_fn is not None else fn

    hlo = static_hlo_check(fn, sharded_args)
    prim = static_primitive_check(fn, sharded_args)
    val = runtime_value_check(lambda _: fn(*sharded_args),
                              lambda _: ref_fn(*rep_args),
                              None, tol=tol)
    return {"hlo": hlo, "primitives": prim, "value": val, "backward": None,
            "static_unsafe": hlo["unsafe"] or bool(prim["suspicious"]),
            "runtime_unsafe": val["unsafe"]}


# ---------------------------------------------------------------------------
# Diagnosis → human-readable messages
# ---------------------------------------------------------------------------
def diagnose(report):
    msgs = []
    v, hlo, prim = report["value"], report["hlo"], report["primitives"]
    if hlo["lowering_error"]:
        msgs.append(("✗", f"this sharding cannot be compiled ({hlo['lowering_error']}); "
                     "replicate the offending array"))
    if v and v.get("replicated_error"):
        msgs.append(("✗", f"replicated reference raised: {v['replicated_error']}"))
    elif v and v["sharded_error"]:
        msgs.append(("✗", f"sharded execution raised: {v['sharded_error']}"))
    elif v and v["match"] is False:
        delta = abs((v["replicated"] or 0.0) - (v["sharded"] or 0.0))
        msgs.append(("✗", f"sharded log-density differs from replicated by {delta:.3g} — "
                     "sharding CHANGES the answer; falling back to replicated"))
    for op, n in hlo["array_collectives"].items():
        sev, text = _COLLECTIVE_MSG.get(op, ("⚠", f"array collective {op}"))
        msgs.append((sev, f"{text} [{op}×{n}]"))
    susp = set(prim["suspicious"])
    if susp & _LINALG:
        msgs.append(("⚠", f"whole-matrix linear algebra ({', '.join(sorted(susp & _LINALG))}) "
                     "on a sharded array — replicate it or exploit structure"))
    if susp & _CROSSPOS:
        msgs.append(("⚠", f"cross-position op(s) ({', '.join(sorted(susp & _CROSSPOS))}) over "
                     "the sharded axis — couples rows across the boundary"))
    if (susp & _INDEX) and not hlo["array_collectives"]:
        msgs.append(("ℹ", f"index/segment op ({', '.join(sorted(susp & _INDEX))}) — GSPMD found "
                     "a local path; correctness depends on index values"))
    back = report.get("backward")
    if back and back.get("dominant"):
        msgs.append(("✗", f"backward-pass communication is O(N²)-dominant "
                     f"(collective ≈{back['coll_elems']:,} elems vs per-shard compute "
                     f"≈{int(back['data_elems']/max(back['n_shards'],1)):,}; "
                     f"ratio {back['ratio']:.2f} ≥ {back['ratio_threshold']:.2f}) — "
                     "the gradient all-reduce of a replicated/coupled parameter does not "
                     "shrink with device count; falling back to replicated"))
    elif back and back.get("coll_elems") is not None and back.get("n_shards", 1) > 1:
        msgs.append(("ℹ", f"backward-pass collective ≈{back['coll_elems']:,} elems "
                     f"(ratio {back['ratio']:.3f} < {back['ratio_threshold']:.2f}) — "
                     "communication-light; sharding can pay off"))
    if not msgs:
        msgs.append(("✓", f"sharding compatible — map+reduce over axis 0, no array "
                     f"collectives (all-reduce×{hlo['reductions']} only)"))
    return msgs


def print_diagnosis(report, name=""):
    print(f"[shard-check{' ' + name if name else ''}]")
    for sev, text in diagnose(report):
        print(f"  {sev} {text}")


# ---------------------------------------------------------------------------
# numpyro integration
# ---------------------------------------------------------------------------
def run_shard_check(model, sharded_kwargs, replicated_kwargs, key, name="", tol=1e-4):
    """Best-effort static + runtime shard check on a numpyro *model*.

    Returns a report dict (with ``static_unsafe`` / ``runtime_unsafe``) or None
    if the check could not be built. Never raises.
    """
    try:
        from numpyro.infer.util import initialize_model

        def _potential(kwargs):
            info = initialize_model(key, model, model_kwargs=kwargs,
                                    dynamic_args=False)
            return info[1], info[0].z   # (potential_fn, init_params)

        pot_s, params = _potential(sharded_kwargs)
        pot_r, _ = _potential(replicated_kwargs)

        hlo = static_hlo_check(pot_s, (params,))
        prim = static_primitive_check(pot_s, (params,))
        val = runtime_value_check(pot_s, pot_r, params, tol=tol)
        back = backward_collective_check(model, params, sharded_kwargs)
        report = {"hlo": hlo, "primitives": prim, "value": val, "backward": back,
                  "static_unsafe": hlo["unsafe"] or bool(prim["suspicious"]),
                  "runtime_unsafe": val["unsafe"],
                  "perf_unsafe": bool(back.get("dominant"))}
        print_diagnosis(report, name)
        return report
    except Exception as e:
        import warnings
        warnings.warn(f"shard-check skipped ({type(e).__name__}: {e})", stacklevel=2)
        return None
