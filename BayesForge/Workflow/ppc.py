"""Posterior predictive calibration and prediction plots for ALL observed
variables in a model -- built on top of ``m.sample()``/``m.model`` (Main/main.py)
and ``Diagnostic/ppc.py``, not a reimplementation of either.

``Diagnostic/ppc.py``'s existing ``get_yrep`` only resolves the *first*
observed variable (``m.obs_args[0]``), with a string-match-or-guess fallback
that isn't reliable for models with more than one ``obs=`` statement. This
module instead traces the model once with ``numpyro.handlers.trace`` and
reads the ``is_observed`` flag directly off every site -- correct regardless
of how many likelihood statements the model has, or whether each one was
given an explicit ``name=``.
"""
import numpy as np

DEFAULT_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def discover_obs_sites(m, seed=0):
    """Return ``{site_name: observed_value_array}`` for every OBSERVED
    (likelihood) site in ``m.model``, found by tracing it once against the
    real ``m.data_on_model``.

    Robust by construction: numpyro's own trace marks a site
    ``is_observed=True`` exactly when it was built with ``obs=<not None>``,
    so this works for any number of ``obs=`` statements and needs no
    heuristic matching between Python variable names and numpyro site names.
    """
    from numpyro import handlers

    if m.model is None or m.data_on_model is None:
        raise RuntimeError("m.model/m.data_on_model not set -- call m.fit() first.")
    trace = handlers.trace(handlers.seed(m.model, rng_seed=seed)).get_trace(**m.data_on_model)
    return {
        name: np.asarray(site["value"])
        for name, site in trace.items()
        if site["type"] == "sample" and site.get("is_observed")
    }


def ppc_replicates(m, samples=500, seed=0, obs_names=None):
    """Draw posterior predictive replicates for every observed site (or just
    ``obs_names`` if given), reusing ``m.sample()`` -- one call, not one per
    observed variable.

    Returns:
        ``(yreps, obs_sites)`` where ``yreps`` is ``{site_name: array(samples, N)}``
        and ``obs_sites`` is ``{site_name: observed_value_array}``
        (from :func:`discover_obs_sites`).
    """
    m.build_model_with_Y_None(m.model)  # ensures m.obs_args is populated (as m.sample() itself does)
    obs_sites = discover_obs_sites(m, seed=seed)
    names = list(obs_names) if obs_names else list(obs_sites.keys())

    # Same pattern Diagnostic/ppc.py's get_yrep already uses: strip the
    # observed arrays out of the data dict ourselves and call with
    # remove_obs=False, since m.sample(remove_obs=True) would otherwise try
    # (and fail) to re-delete keys already absent from a pre-stripped dict.
    pred_data = {k: v for k, v in m.data_on_model.items() if k not in (m.obs_args or [])}
    pred = m.sample(data=pred_data, remove_obs=False, posterior=True,
                    samples=samples, seed=seed, return_sites=names)

    yreps = {}
    for name in names:
        if name not in pred:
            continue
        arr = np.asarray(pred[name])
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        yreps[name] = arr
    return yreps, {k: v for k, v in obs_sites.items() if k in yreps}


def quantile_calibration_table(m, samples=500, levels=DEFAULT_LEVELS,
                                max_count=None, seed=0, obs_names=None):
    """Bayesian-p-value calibration, for every observed variable, auto-picking
    the right check per variable from its dtype:

    - **continuous** (float-valued observations): Bayesian p-value at each
      quantile ``level`` -- ``P(replicated quantile > observed quantile)``.
      Well-calibrated means each p-value sits near 0.5 (systematically near 0
      or 1 means the model's predictive distribution is shifted/too
      narrow/too wide at that quantile).
    - **discrete/count** (integer-valued observations): same idea applied to
      the empirical CDF at each integer count 0..max_count instead of a
      continuous quantile.

    Args:
        m: A fitted ``bf`` instance.
        samples: Posterior predictive draws to use.
        levels: Quantile levels for continuous observed variables.
        max_count: Highest count checked for discrete variables (default:
            each variable's own observed max).
        seed: Passed to ``m.sample()``.
        obs_names: Restrict to these observed variables (default: all of them).

    Returns:
        A long-format pandas DataFrame: columns ``obs`` (variable/site name),
        ``kind`` (``"quantile"`` or ``"count"``), ``level`` (the quantile
        level or count), ``p_value``.
    """
    import pandas as pd

    yreps, obs_sites = ppc_replicates(m, samples=samples, seed=seed, obs_names=obs_names)
    rows = []
    for name, y_obs in obs_sites.items():
        yrep = yreps[name]
        y_obs = np.asarray(y_obs).reshape(-1)

        if np.issubdtype(y_obs.dtype, np.integer):
            hi = int(max_count) if max_count is not None else int(np.max(y_obs))
            counts = np.arange(0, hi + 1)
            obs_cdf = np.array([(y_obs <= k).mean() for k in counts])
            rep_cdf = np.stack([(yrep <= k).mean(axis=1) for k in counts])  # (n_counts, samples)
            p = (rep_cdf > obs_cdf[:, None]).mean(axis=1)
            for k, pv in zip(counts, p):
                rows.append(dict(obs=name, kind="count", level=int(k), p_value=float(pv)))
        else:
            lv = np.asarray(levels, dtype=float)
            obs_q = np.quantile(y_obs, lv)
            rep_q = np.quantile(yrep, lv, axis=1)  # (n_levels, samples)
            p = (rep_q > obs_q[:, None]).mean(axis=1)
            for l, pv in zip(lv, p):
                rows.append(dict(obs=name, kind="quantile", level=float(l), p_value=float(pv)))

    return pd.DataFrame(rows)


def compare_calibration_table(models, samples=500, levels=DEFAULT_LEVELS,
                               max_count=None, seed=0, obs_names=None):
    """:func:`quantile_calibration_table` for several fitted models at once,
    stacked with a ``model`` column -- for the "does the calibration check
    agree between my full and reduced model" comparison.

    Args:
        models: ``dict[label -> fitted bf instance]``.
        Others: forwarded to :func:`quantile_calibration_table` for each model.

    Returns:
        A long-format pandas DataFrame like :func:`quantile_calibration_table`
        with an added ``model`` column.
    """
    import pandas as pd

    frames = []
    for label, model_m in models.items():
        t = quantile_calibration_table(model_m, samples=samples, levels=levels,
                                       max_count=max_count, seed=seed, obs_names=obs_names)
        t["model"] = label
        frames.append(t)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["obs", "kind", "level", "p_value", "model"])
