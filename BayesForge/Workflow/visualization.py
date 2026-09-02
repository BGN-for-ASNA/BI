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
