"""Inject fixed parameter values into a DGP's ``sample=True`` draws --
without requiring the DGP to declare anything special.

``BayesForge.Distributions.np_dists.DistProxy`` already auto-infers the
``name=`` kwarg for every ``m.dist.<dist>(...)`` call from the assignment's
LHS variable (bytecode inspection -- see ``_infer_caller_name`` there), so
``beta = m.dist.normal(0, 1, sample=True)`` already implicitly carries
``name="beta"`` today, with nothing typed. This module hooks in *below* that
inference: it swaps ``m.dist``'s underlying distribution object for a
transparent wrapper that, for any call whose (already-inferred-by-BF, or
explicit) ``name`` matches a key in the fixed-value dict, returns that fixed
value instead of sampling -- and otherwise passes straight through to the
real distribution unchanged.

Net effect: ``dgp()`` is called exactly as written (``dgp()``, no new
argument), and existing idiomatic DGP code needs zero edits to become
sweepable by ``Workflow.recover(param_grid=...)``.
"""
import functools
import inspect
from contextlib import contextmanager

from BayesForge.Distributions.np_dists import _has_name_kwarg


class _FixedValueDist:
    """Wraps a real ``UnifiedDist`` instance. Distribution calls whose
    ``sample=True`` and whose ``name`` is a key in ``fixed`` return the fixed
    value (broadcast to ``shape`` if given) instead of sampling; everything
    else passes through to the real object unchanged.
    """

    def __init__(self, real_dist_obj, fixed):
        object.__setattr__(self, "_real", real_dist_obj)
        object.__setattr__(self, "_fixed", fixed)

    def __getattr__(self, attr):
        real = object.__getattribute__(self, "_real")
        fixed = object.__getattribute__(self, "_fixed")
        target = getattr(real, attr)
        if not callable(target):
            return target
        func = getattr(target, "__func__", target)
        if not _has_name_kwarg(func):
            return target

        @functools.wraps(target)
        def wrapper(*args, **kwargs):
            import jax.numpy as jnp

            name = kwargs.get("name")
            if kwargs.get("sample") and name is not None and name in fixed:
                value = jnp.asarray(fixed[name])
                shape = kwargs.get("shape") or ()
                return jnp.broadcast_to(value, shape) if shape else value
            return target(*args, **kwargs)

        return wrapper

    def __setattr__(self, attr, value):
        setattr(object.__getattribute__(self, "_real"), attr, value)


@contextmanager
def fixed_params(m, fixed):
    """Temporarily override ``m.dist``'s ``sample=True`` draws named in
    ``fixed`` with fixed values, for the duration of the ``with`` block.

    Args:
        m: A ``bf`` instance.
        fixed: ``dict[name -> value]``. Empty/None is a no-op (real sampling
            proceeds unchanged -- so callers don't need to special-case an
            empty grid row).

    Example:
        >>> with fixed_params(m, {"beta": -2.0}):
        ...     true_params, data = dgp()   # dgp() called exactly as always;
        ...     # true_params["beta"] == -2.0, and `data` was generated
        ...     # from that exact value, not a random draw.
    """
    if not fixed:
        yield
        return
    dist_proxy = m.dist
    real_dist = object.__getattribute__(dist_proxy, "_dist")
    object.__setattr__(dist_proxy, "_dist", _FixedValueDist(real_dist, fixed))
    try:
        yield
    finally:
        object.__setattr__(dist_proxy, "_dist", real_dist)


def split_grid_row(dgp, row):
    """Split one ``param_grid`` row into (call_kwargs, sample_fixed).

    A ``param_grid`` key can target two different things, and this figures
    out which without the DGP declaring anything special either way:

    - a plain **argument of ``dgp`` itself** (e.g. ``def dgp(n_obs=200):``,
      for sweeping sample size / a fixed design setting rather than a
      sampled parameter) -- detected via ``inspect.signature(dgp)`` and
      returned in ``call_kwargs``, to be passed as ``dgp(**call_kwargs)``.
    - an **internal ``sample=True`` draw** (e.g. ``beta = m.dist.normal(...,
      sample=True)``) -- everything else, returned in ``sample_fixed``, to be
      applied via :func:`fixed_params` around the call.

    A name present in both (an argument AND a matching internal draw name)
    is treated as a call argument -- the more direct route -- and left out
    of ``sample_fixed``.

    Args:
        dgp: The DGP callable for this row (used only for its signature).
        row: ``dict[name -> value]`` for one simulation (already indexed by
            simulation number by the caller).

    Returns:
        ``(call_kwargs, sample_fixed)``, both plain dicts.
    """
    if not row:
        return {}, {}
    try:
        sig_params = inspect.signature(dgp).parameters
    except (TypeError, ValueError):
        sig_params = {}
    call_kwargs = {k: v for k, v in row.items() if k in sig_params}
    sample_fixed = {k: v for k, v in row.items() if k not in sig_params}
    return call_kwargs, sample_fixed
