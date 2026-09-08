"""Plots for Workflow results.

Uses plotly, matching the convention already established in
``Diagnostic/Diag2.py``/``patch_diag.py`` (``m.diag.plot_trace()``,
``m.diag.forest()``, etc.), so a ``Workflow`` figure looks and behaves like
the rest of BF's diagnostic plots rather than introducing a second plotting
style.
"""


def plot_recovery(result, param_names=None):
    """Estimated-vs-true scatter with HDI error bars, one panel per parameter.

    Points are colored green when the HDI covers the true value and red when
    it doesn't, with a dashed 1:1 reference line -- the standard parameter
    recovery plot.

    Args:
        result: A :class:`~BayesForge.Workflow.results.RecoveryResult`.
        param_names: Subset of parameters to plot (default: all of them).

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    names = param_names or result.param_names
    fig = make_subplots(rows=1, cols=len(names), subplot_titles=names)

    for i, name in enumerate(names):
        table = result.table
        true_vals = table[f"{name}_true"]
        est_vals = table[f"{name}_mean"]
        hdi_lo = table[f"{name}_hdi_lo"]
        hdi_hi = table[f"{name}_hdi_hi"]
        covered = table[f"{name}_covered"]
        colors = ["seagreen" if c else "firebrick" for c in covered]

        lo = float(min(true_vals.min(), est_vals.min()))
        hi = float(max(true_vals.max(), est_vals.max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 line=dict(dash="dash", color="gray"),
                                 showlegend=False), row=1, col=i + 1)
        fig.add_trace(go.Scatter(
            x=true_vals, y=est_vals, mode="markers",
            marker=dict(color=colors, size=7),
            error_y=dict(type="data", symmetric=False,
                        array=hdi_hi - est_vals, arrayminus=est_vals - hdi_lo,
                        width=2, thickness=1),
            showlegend=False,
        ), row=1, col=i + 1)

        r2 = result.metrics[name]["r2"]
        grade = result.metrics[name]["grade"]
        fig.update_xaxes(title_text=f"true {name}", row=1, col=i + 1)
        fig.update_yaxes(title_text=f"estimated {name}", row=1, col=i + 1)
        fig.layout.annotations[i].text = f"{name} (R²={r2:.2f}, grade {grade})"

    fig.update_layout(title_text="Parameter recovery", height=420,
                      width=max(420, 380 * len(names)), plot_bgcolor="white")
    return fig


def plot_sbc_rank(result, param_names=None, bins=None):
    """Rank histograms, one panel per parameter -- flat is well-calibrated.

    Args:
        result: A :class:`~BayesForge.Workflow.results.SBCResult`.
        param_names: Subset of parameters to plot (default: all of them).
        bins: Histogram bin count (default: ``uniformity[name]["n_bins"]``).

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    names = param_names or result.param_names
    fig = make_subplots(rows=1, cols=len(names), subplot_titles=names)

    for i, name in enumerate(names):
        ranks = result.table[f"{name}_rank"]
        n_bins = bins or result.uniformity[name]["n_bins"]
        p_value = result.uniformity[name]["p_value"]
        color = "#2ca02c" if (p_value == p_value and p_value > 0.05) else "#d62728"
        fig.add_trace(go.Histogram(x=ranks, nbinsx=n_bins, marker_color=color,
                                   showlegend=False), row=1, col=i + 1)
        fig.update_xaxes(title_text=f"rank of true {name}", row=1, col=i + 1)
        fig.layout.annotations[i].text = f"{name} (p={p_value:.3f})"

    fig.update_layout(title_text="SBC rank histograms (flat = well-calibrated)",
                      height=380, width=max(380, 340 * len(names)),
                      plot_bgcolor="white")
    return fig


def plot_annotated_summary(table):
    """Render an :func:`~BayesForge.Workflow.diagnostics.annotated_summary`
    table as a plotly Table, with rows colored by verdict.

    Args:
        table: The DataFrame returned by ``diagnostics.annotated_summary``
            (must have a ``verdict`` column).

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    import plotly.graph_objects as go

    verdict_colors = {"OK": "#e6f4ea", "CHECK": "#fff4e5", "POOR": "#fdecea"}
    display = table.reset_index().rename(columns={"index": "parameter"})
    row_colors = [verdict_colors.get(v, "#ffffff") for v in display["verdict"]]

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(display.columns), fill_color="#f0f0f0",
                   align="left"),
        cells=dict(values=[display[c] for c in display.columns],
                  fill_color=[row_colors] * len(display.columns),
                  align="left"),
    )])
    fig.update_layout(title_text="Annotated posterior summary")
    return fig


def plot_calibration(table, band=(0.1, 0.9)):
    """Bayesian-p-value calibration plot from
    :func:`~BayesForge.Workflow.ppc.quantile_calibration_table` (or
    :func:`~BayesForge.Workflow.ppc.compare_calibration_table`) -- one panel
    per observed variable, y-axis the Bayesian p-value at each quantile
    level / count, shaded acceptable band, dashed line at the ideal (0.5).

    If ``table`` has a ``model`` column (from ``compare_calibration_table``),
    each model gets its own line/color per panel, matching the R
    ``quantile_calibration`` example this mirrors.

    Args:
        table: DataFrame with columns ``obs``, ``kind``, ``level``,
            ``p_value`` (and optionally ``model``).
        band: (lo, hi) shaded "acceptable" p-value range, default (0.1, 0.9).

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    has_model = "model" in table.columns
    panels = list(table[["obs", "kind"]].drop_duplicates().itertuples(index=False, name=None))
    if not panels:
        return go.Figure()
    colors = ["#2a78d6", "#eb6834", "#2ca02c", "#9467bd", "#d62728"]

    fig = make_subplots(rows=1, cols=len(panels),
                        subplot_titles=[f"{obs} ({kind})" for obs, kind in panels])
    for col, (obs_name, kind) in enumerate(panels, start=1):
        sub = table[(table["obs"] == obs_name) & (table["kind"] == kind)].sort_values("level")
        fig.add_hrect(y0=band[0], y1=band[1], fillcolor="#f0efec", opacity=0.6,
                      line_width=0, row=1, col=col)
        fig.add_hline(y=0.5, line_dash="dash", line_color="#898781", row=1, col=col)

        if has_model:
            for i, (label, grp) in enumerate(sub.groupby("model", sort=False)):
                c = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=grp["level"].astype(str), y=grp["p_value"], mode="lines+markers",
                    name=str(label), legendgroup=str(label), showlegend=(col == 1),
                    line=dict(color=c), marker=dict(color=c, size=8),
                ), row=1, col=col)
        else:
            fig.add_trace(go.Scatter(
                x=sub["level"].astype(str), y=sub["p_value"], mode="lines+markers",
                showlegend=False, line=dict(color=colors[0]), marker=dict(color=colors[0], size=8),
            ), row=1, col=col)

        fig.update_yaxes(range=[0, 1], title_text="Bayesian p-value" if col == 1 else None,
                         row=1, col=col)
        fig.update_xaxes(title_text=kind, row=1, col=col)

    fig.update_layout(
        title_text="Posterior predictive calibration",
        height=420, width=max(420, 380 * len(panels)),
        plot_bgcolor="white", legend_title_text="model" if has_model else None,
    )
    return fig


def plot_predictions(m, samples=500, seed=0, prob=0.5, prob_outer=0.9, obs_names=None):
    """Prediction plot for EVERY observed variable in the model: predicted
    median + interval ribbon vs. observation index, with actual observed
    values overlaid.

    Reuses ``Diagnostic/ppc.py``'s existing ``ppc_ribbon`` per observed
    variable rather than reimplementing the interval-band drawing -- this
    function's job is only finding every observed variable
    (:func:`~BayesForge.Workflow.ppc.ppc_replicates`) and calling that
    existing function once per variable.

    Args:
        m: A fitted ``bf`` instance.
        samples: Posterior predictive draws.
        seed: Passed to ``m.sample()``.
        prob, prob_outer: Inner/outer interval mass, forwarded to ``ppc_ribbon``.
        obs_names: Restrict to these observed variables (default: all of them).

    Returns:
        ``dict[obs_name -> plotly.graph_objects.Figure]`` -- one figure per
        observed variable (they may have unrelated x-axes/scales, so these
        are kept separate rather than force-faceted into one figure).
    """
    from BayesForge.Workflow.ppc import ppc_replicates
    from BayesForge.Diagnostic.ppc import ppc_ribbon
    import numpy as np

    yreps, obs_sites = ppc_replicates(m, samples=samples, seed=seed, obs_names=obs_names)
    figs = {}
    for name, y_obs in obs_sites.items():
        y_obs = np.asarray(y_obs).reshape(-1)
        x = np.arange(len(y_obs))
        figs[name] = ppc_ribbon(y_obs, yreps[name], x=x, prob=prob, prob_outer=prob_outer,
                                title=f"Posterior predictive: {name}")
    return figs


def save_figure(fig, path):
    """Write a plotly figure to a static image file (PNG/PDF/SVG).

    Requires the ``kaleido`` package for static export; warns (does not
    raise) if it's unavailable, so a missing optional dependency never
    crashes a ``Workflow.recover``/``sbc`` run's ``results_dir`` export.
    """
    try:
        fig.write_image(path)
    except Exception as e:  # pragma: no cover - depends on optional kaleido
        import warnings
        warnings.warn(f"Could not save figure to {path} ({e}). "
                      "Install the 'kaleido' package for static image export "
                      "(`pip install kaleido`).")
