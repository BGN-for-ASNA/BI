"""Parallel execution backing the Workflow class's multi-fit loops.

BF's confirmed parallelization surface (verified against ``Main/main.py`` and
``SetDevice/set.py``, not just the docs) has two independent layers:

1. **Chain-level, always available** -- ``m.fit(..., num_chains=4,
   chain_method="parallel")`` is NumPyro's own parallel-chains execution;
   ``bf(cores=N)`` sets up N virtual XLA devices on CPU so those chains
   actually run concurrently rather than time-sliced. This is BF's default
   and every :meth:`Workflow.recover`/:meth:`Workflow.sbc` fit inherits it
   automatically via ``fit_kwargs``.
2. **Multi-device data sharding, opt-in** -- on a multi-GPU (or
   ``cores>1`` CPU) setup, ``m.fit(..., shard=True)`` splits the leading axis
   of observation arrays across devices (``m.shard()``/``m.replicate()``),
   with correctness safeguards in ``Diagnostic/sharding_safeguards.py``. This
   parallelizes *within* one fit's likelihood evaluation, not across fits.

Neither layer parallelizes the *outer* loop this module adds: running N
independent simulate-then-fit replications (recovery, SBC). A BF model is a
Python closure over one specific ``bf`` instance (``m.dist.normal(...)``
inside the model function captures that ``m``), so those closures cannot be
pickled to a worker process. Genuine process-level parallelism for the outer
loop therefore requires each worker to build its *own* ``bf`` instance from a
factory function rather than reusing a shared closure -- see
``model_factory``/``dgp_factory`` in ``core.py``.
"""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def probe_parallel_capabilities(m=None, verbose=True):
    """Report what parallel execution is actually available right now.

    Args:
        m: An existing ``bf`` instance (optional). When given, its
            ``n_devices``/``n_gpus``/sharding state is included.
        verbose: Print a short human-readable report.

    Returns:
        dict with keys: ``jax_backend``, ``local_device_count``, ``cpu_count``,
        ``chain_method_default`` (``"parallel"``, matching ``bf.fit``'s
        default), ``n_devices``, ``n_gpus``, ``sharding_available`` (the
        latter three only when ``m`` is given).
    """
    import jax

    report = {
        "jax_backend": jax.default_backend(),
        "local_device_count": jax.local_device_count(),
        "cpu_count": os.cpu_count(),
        "chain_method_default": "parallel",
    }
    if m is not None:
        report["n_devices"] = getattr(m, "n_devices", None)
        report["n_gpus"] = getattr(m, "n_gpus", None)
        report["sharding_available"] = getattr(m, "_data_sharding", None) is not None

    if verbose:
        lines = [
            "BF parallelization capabilities:",
            f"  JAX backend:            {report['jax_backend']}",
            f"  Local JAX devices:      {report['local_device_count']}",
            f"  CPU cores (os):         {report['cpu_count']}",
            f"  m.fit default chains:   num_chains=4, chain_method='parallel'",
        ]
        if m is not None:
            lines.append(f"  m.n_devices:            {report['n_devices']}")
            lines.append(f"  m.n_gpus:               {report['n_gpus']}")
            lines.append(f"  Multi-device sharding:  "
                         f"{'available' if report['sharding_available'] else 'not active (n_devices<=1)'}")
        print("\n".join(lines))
    return report


def _sim_worker(i, model_factory, dgp_factory, param_names, fit_kwargs,
                 platform, cores, hdi_prob, mode):
    """Runs inside its own process: build a fresh ``bf`` instance, simulate one
    dataset, fit it, and reduce the fit down to plain-Python numbers so the
    result can be pickled back to the parent process.

    Not imported/used directly -- dispatched via :func:`run_parallel`.
    """
    from BayesForge import bf
    import jax.numpy as jnp

    m = bf(platform=platform, cores=cores, print_devices_found=False)
    model = model_factory(m)
    dgp = dgp_factory(m)
    true_params, data = dgp()
    m.fit(model, obs=data, **fit_kwargs)

    row = {"sim": i}
    if mode == "recover":
        lo_q, hi_q = (1 - hdi_prob) / 2, 1 - (1 - hdi_prob) / 2
        for name in param_names:
            draws = jnp.asarray(m.posteriors[name]).reshape(-1)
            lo, hi = jnp.quantile(draws, jnp.array([lo_q, hi_q]))
            true_val = float(true_params[name])
            row[f"{name}_true"] = true_val
            row[f"{name}_mean"] = float(draws.mean())
            row[f"{name}_hdi_lo"] = float(lo)
            row[f"{name}_hdi_hi"] = float(hi)
            row[f"{name}_covered"] = bool(lo <= true_val <= hi)
    else:  # mode == "sbc"
        for name in param_names:
            draws = jnp.asarray(m.posteriors[name]).reshape(-1)
            true_val = float(true_params[name])
            row[f"{name}_rank"] = int((draws < true_val).sum())
            row[f"{name}_n_draws"] = int(draws.size)
    return row


def run_parallel(n_sim, n_jobs=1, **worker_kwargs):
    """Dispatch ``n_sim`` independent :func:`_sim_worker` calls.

    Args:
        n_sim: Number of independent replications.
        n_jobs: 1 (default) runs sequentially in-process (no new ``bf``
            instances are created here -- see ``core.py`` for the cheaper
            shared-instance path used when n_jobs=1). >1 spawns a
            ``ProcessPoolExecutor`` with that many workers, each building its
            own ``bf`` instance via ``model_factory``/``dgp_factory``.
        **worker_kwargs: Forwarded to :func:`_sim_worker` (model_factory,
            dgp_factory, param_names, fit_kwargs, platform, cores, hdi_prob,
            mode).

    Returns:
        list of per-simulation result dicts, in simulation order.
    """
    if n_jobs <= 1:
        return [_sim_worker(i, **worker_kwargs) for i in range(n_sim)]

    rows = [None] * n_sim
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {ex.submit(_sim_worker, i, **worker_kwargs): i for i in range(n_sim)}
        for fut in as_completed(futures):
            i = futures[fut]
            rows[i] = fut.result()
    return rows
