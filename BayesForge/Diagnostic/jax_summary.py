"""DEPRECATED. Use :mod:`BayesForge.Diagnostic.jax_diagnostics` instead.

This module used to carry its own summary statistics. Every estimator in it was
wrong, and nothing in the package imported it (``main.py`` uses
``jax_diagnostics.summary``, aliased ``_jax_summary``, which is a different
module). The old implementations:

  * ``calculate_r_hat``  — plain Gelman-Rubin: neither split nor
    rank-normalized, so it could not detect within-chain non-stationarity at
    all (1.0014 where ArviZ reported 1.0071).
  * ``calculate_ess``    — the Geyer pair sum started at (rho_2, rho_3) and so
    omitted rho_1, the largest autocorrelation term, over-estimating ESS by
    ~10% on an AR(0.9) chain (437.9 vs 396.6). It also applied Geyer's
    positivity rule per chain after clamping, and never normalized by the
    combined within+between variance.
  * ``ess_tail``         — was literally a copy of ``ess_bulk``.
  * ``hdi_5.5%`` / ``hdi_94.5%`` — held equal-tailed percentiles, not an HDI.

Everything here now forwards to the ArviZ-parity implementations. Keep using it
only until callers are migrated; it will be removed.
"""
import warnings

import numpy as np
import pandas as pd

from BayesForge.Diagnostic.jax_diagnostics import (
    _ess_1d,
    _ess_tail_1d,
    _rhat_1d,
    summary as _summary,
)

__all__ = ["calculate_r_hat", "calculate_ess", "jax_summary"]

_MSG = ("BayesForge.Diagnostic.jax_summary is deprecated and forwards to "
        "BayesForge.Diagnostic.jax_diagnostics; import from there instead.")


def calculate_r_hat(chains) -> float:
    """Rank-normalized folded-split R-hat. chains: (n_chains, n_draws).

    Delegates to ``jax_diagnostics._rhat_1d``; matches ``az.rhat(method="rank")``.
    """
    warnings.warn(_MSG, DeprecationWarning, stacklevel=2)
    return _rhat_1d(np.asarray(chains))


def calculate_ess(chains) -> float:
    """Rank-normalized split bulk ESS. chains: (n_chains, n_draws).

    Delegates to ``jax_diagnostics._ess_1d``; matches ``az.ess(method="bulk")``.
    """
    warnings.warn(_MSG, DeprecationWarning, stacklevel=2)
    return _ess_1d(np.asarray(chains))


def calculate_ess_tail(chains) -> float:
    """Tail ESS. Previously this was a copy of the bulk value."""
    warnings.warn(_MSG, DeprecationWarning, stacklevel=2)
    return _ess_tail_1d(np.asarray(chains))


def jax_summary(posterior_dict: dict) -> pd.DataFrame:
    """Summary statistics for a dict of chain-major posterior samples.

    Delegates to ``jax_diagnostics.summary``, whose columns match
    ``az.summary``. Indexed by ``var`` and sorted, as this function always was.
    """
    warnings.warn(_MSG, DeprecationWarning, stacklevel=2)
    normalized = {
        k: (np.asarray(v)[None, :] if np.asarray(v).ndim == 1 else np.asarray(v))
        for k, v in posterior_dict.items()
    }
    df = _summary(normalized, round_to=3, hdi_prob=0.89, group_by_chain=True)
    return df.rename_axis("var").sort_index()
