"""Re-export the ArviZ-backed diagnostics from their canonical location.

BF's diagnostics live in the package (``main.py`` imports them, so they cannot
live under ``Test/``). This module follows the same shim convention as
``Test/Diagnostic/summary/patch_diag.py`` and gives the diagnostic test
subfolder a single import point.

Layout
------
``BayesForge.Diagnostic.Diag.diag``
    The ArviZ-backed class: ``az.summary`` / ``az.rhat`` / ``az.ess`` /
    ``az.loo``. This is the reference implementation and the default engine.

``BayesForge.Diagnostic.Diag2.diagWIP``
    The class ``m.diag`` is built from. Adds ``diagnose()`` (Stan-style
    convergence report) on top of the ArviZ delegation.

``BayesForge.Diagnostic.patch_diag.bind_diag_to_model(diag_obj, m, backend=)``
    Installs the live methods on ``m.diag``. ``backend="arviz"`` (default)
    routes the convergence metrics to the class above; ``backend="jax"``
    routes them to ``jax_diagnostics`` and warns that it is experimental.
    Plots, PPC, sensitivity and regression overlays are backend-independent.

Backend selection
-----------------
    m = bf()                          # arviz (default)
    m = bf(diag_backend="jax")        # experimental; warns

    m.summary()                       # arviz
    m.summary(backend="jax")          # experimental; warns
    m.summary_jax()                   # same, explicit
"""
from BayesForge.Diagnostic.Diag import diag                      # noqa: F401
from BayesForge.Diagnostic.Diag2 import diagWIP                  # noqa: F401
from BayesForge.Diagnostic.patch_diag import (                   # noqa: F401
    ARVIZ,
    JAX,
    bind_diag_to_model,
    patch_diag_class,
    _resolve_backend,
    _az_diag_for,
)

#: The ArviZ-backed diag class, under a name that says which engine it uses.
DiagArviZ = diag

__all__ = [
    "diag",
    "DiagArviZ",
    "diagWIP",
    "bind_diag_to_model",
    "patch_diag_class",
    "ARVIZ",
    "JAX",
]
