"""Diagnostics that ``m.summary()``/``m.diag`` don't already provide.

``m.summary()`` (``Diagnostic/jax_diagnostics.py``) returns raw numbers --
mean, sd, hdi, mcse_mean, mcse_sd, ess_bulk, ess_tail, r_hat -- and
``m.diag.diagnose()`` (``Diagnostic/Diag2.py``) returns a separate free-text
sampler-level report (treedepth, divergences, E-BFMI). Neither attaches a
per-parameter, human-readable verdict to the summary table itself. This
module adds that: ``annotated_summary`` merges ``m.summary()`` with an
``interpretation``/``verdict`` column per parameter. It deliberately does not
duplicate ``m.diag.diagnose()``'s sampler-wide checks -- call both.
"""
import inspect


def annotated_summary(m, round_to=4, hdi_prob=0.89,
                       rhat_threshold=1.01, ess_threshold=400,
                       mcse_frac_threshold=0.1,
                       include=None, exclude=None):
    """``m.summary()`` with an added ``verdict``/``interpretation`` column.

    Args:
        m: A fitted ``bf`` instance (``m.fit(...)`` already called).
        round_to: Decimal places, forwarded to ``m.summary()``.
        hdi_prob: HDI mass, forwarded to ``m.summary()``.
        rhat_threshold: R-hat above this is flagged (BF/Stan convention: 1.01).
        ess_threshold: ess_bulk or ess_tail below this is flagged (400).
        mcse_frac_threshold: mcse_mean / sd above this fraction is flagged --
            i.e. Monte Carlo noise is a non-trivial share of the posterior
            spread, so extra reported digits would be simulation noise.
        include, exclude: Forwarded to ``m.summary()``.

    Returns:
        The same pandas DataFrame ``m.summary()`` returns, with two added
        columns: ``verdict`` (``"OK"``/``"CHECK"``/``"POOR"``) and
        ``interpretation`` (the reason(s), or a confirmation when OK).
    """
    table = m.summary(round_to=round_to, hdi_prob=hdi_prob,
                       include=include, exclude=exclude)
    table = table.copy()

    verdicts, notes = [], []
    for _, row in table.iterrows():
        flags = []

        rhat = row.get("r_hat")
        if rhat is not None and rhat == rhat and rhat > rhat_threshold:
            flags.append(f"R-hat {rhat:.3f} > {rhat_threshold}: chains disagree "
                         "-- run more chains/draws or reparameterize (non-centered)")

        ess_bulk = row.get("ess_bulk")
        if ess_bulk is not None and ess_bulk == ess_bulk and ess_bulk < ess_threshold:
            flags.append(f"ess_bulk {ess_bulk:.0f} < {ess_threshold}: mean/sd unreliable "
                         "-- increase num_samples")

        ess_tail = row.get("ess_tail")
        if ess_tail is not None and ess_tail == ess_tail and ess_tail < ess_threshold:
            flags.append(f"ess_tail {ess_tail:.0f} < {ess_threshold}: HDI/tail quantiles "
                         "unreliable -- increase num_samples")

        sd = row.get("sd")
        mcse_mean = row.get("mcse_mean")
        if (sd is not None and mcse_mean is not None and sd == sd and mcse_mean == mcse_mean
                and sd > 0 and (mcse_mean / sd) > mcse_frac_threshold):
            frac = mcse_mean / sd
            flags.append(f"mcse_mean is {frac:.0%} of sd: report fewer significant "
                         "digits, or draw more samples to shrink it")

        if not flags:
            verdicts.append("OK")
            notes.append("Converged: R-hat/ESS/MCSE within thresholds.")
        elif len(flags) == 1:
            verdicts.append("CHECK")
            notes.append(flags[0])
        else:
            verdicts.append("POOR")
            notes.append(" | ".join(flags))

    table["verdict"] = verdicts
    table["interpretation"] = notes
    return table


def advise(m=None, data=None, n_params=None, dgp=None, model=None,
           n_obs_svi_threshold=50_000, n_params_svi_threshold=200):
    """Best-effort checklist for DGP conventions and MCMC-vs-SVI choice.

    This is a heuristic pass over what's actually knowable up front (data
    size, and -- when source is inspectable -- the DGP/model source text). It
    is not a substitute for ``m.diag.diagnose()`` after fitting; it is meant
    to catch avoidable mistakes *before* spending a full MCMC run.

    Args:
        m: A ``bf`` instance (optional, unused for now beyond future
            extension -- accepted for a consistent call signature with the
            rest of the class).
        data: The ``obs`` dict about to be passed to ``m.fit`` -- used to
            estimate N for the inference-mode recommendation.
        n_params: Optional explicit parameter count (if known) -- used for
            the inference-mode recommendation alongside data size.
        dgp: Optional DGP callable to source-inspect for common convention
            violations (Python for-loops instead of ``shape=``, missing
            ``sample=True``, fixed-X via ``linspace`` where a genuine
            simulation study needs random X). Best-effort: silently skipped
            if the source isn't inspectable (e.g. defined in a REPL).
        model: Optional model callable, inspected the same way as ``dgp``
            (checked for Python for-loops over ``m.dist.*``).
        n_obs_svi_threshold: N above which SVI is suggested for a first pass.
        n_params_svi_threshold: latent-parameter count above which SVI is
            suggested for a first pass.

    Returns:
        list[str] of advisory messages (empty list means nothing flagged).
    """
    advice = []

    # --- Inference mode: MCMC (m.fit) vs SVI (m.svi) --------------------
    n_obs = None
    if data:
        for v in data.values():
            shape = getattr(v, "shape", None)
            if shape:
                n_obs = max(n_obs or 0, shape[0])
    big_data = n_obs is not None and n_obs > n_obs_svi_threshold
    big_model = n_params is not None and n_params > n_params_svi_threshold
    if big_data or big_model:
        advice.append(
            "Large problem (N="
            f"{n_obs if n_obs is not None else '?'}, params="
            f"{n_params if n_params is not None else '?'}): consider `m.svi(model, "
            "guide='multivariate')` for a fast approximate fit to screen model "
            "variants before committing to full `m.fit()` NUTS/HMC."
        )
    elif n_obs is not None:
        advice.append(
            f"N={n_obs} is modest: `m.fit()` (NUTS/HMC) should be directly "
            "tractable. Reserve `m.svi()` for quick screening of structural "
            "model changes, not as the final inference."
        )

    # --- DGP / model source inspection (best-effort) ---------------------
    for label, fn in (("dgp", dgp), ("model", model)):
        if fn is None:
            continue
        try:
            src = fn if isinstance(fn, str) else inspect.getsource(fn)
        except (OSError, TypeError):
            continue
        if "for " in src and ("m.dist." in src or "range(" in src):
            advice.append(
                f"{label}: contains a Python for-loop near `m.dist.*` calls -- "
                "BF's convention is vectorised `shape=(N,)` sampling; a "
                "for-loop over m.dist.* is both slow and against the "
                "documented DGP pattern."
            )
        if label == "dgp" and "sample=True" not in src:
            advice.append(
                "dgp: no `sample=True` found -- if this function is meant to "
                "simulate data (not declare a model), every stochastic draw "
                "needs sample=True or it will try to register a latent site "
                "instead of returning a value."
            )
        if label == "dgp" and ("linspace" in src) and ("sample=True" in src):
            advice.append(
                "dgp: predictor built with linspace alongside sampled "
                "parameters -- fine for an EXPERIMENTAL/fixed-X design, but a "
                "genuine simulation study usually needs X itself drawn from a "
                "distribution (`m.dist.normal(..., sample=True, shape=(N,))`) "
                "-- see the observational-vs-experimental DGP distinction in "
                "bf://how-to/python/data-generation."
            )

    if not advice:
        advice.append("No issues detected from the information given.")
    return advice
