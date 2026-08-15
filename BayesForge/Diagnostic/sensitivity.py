"""Posterior sensitivity analysis and model checking utilities for BF.

Functions
---------
prior_sensitivity_plot   — overlay posteriors fitted with different priors
influence_plot           — Pareto-k scatter (LOO influential observations)
calibration_plot         — empirical coverage vs nominal level
divergence_energy_plot   — HMC energy / divergence check
multimodality_check      — cluster-based mode detection
"""
import numpy as np
import jax.numpy as jnp
import plotly.graph_objects as go
import plotly.colors as pcolors
from plotly.subplots import make_subplots
import plotly.express as px

_COLORS = pcolors.qualitative.Plotly


# =============================================================================
# 1. Prior sensitivity
# =============================================================================

def prior_sensitivity_plot(
    posteriors_list,
    labels,
    param_names=None,
    hdi_prob=0.89,
    title="Prior Sensitivity Analysis",
):
    """Overlay posteriors from models fitted with different priors/assumptions.

    Compares posterior distributions across multiple model fits to assess
    how much conclusions change when priors (or likelihood assumptions) vary.

    Args:
        posteriors_list: List of posterior dicts, each {param: array(S,)}.
                         Typically [m1.posteriors, m2.posteriors, ...].
        labels: List of labels for each fit (e.g. ['weakly informative',
                'strong prior', 'alternative likelihood']).
        param_names: Subset of parameters to compare. Defaults to all
                     parameters present in all fits.
        hdi_prob: HDI probability for interval markers.
        title: Plot title.

    Returns:
        plotly Figure with one subplot per parameter.

    Example:
        # Refit with a tighter prior on b and compare:
        m1.fit(model_weak, ...); m2.fit(model_strong, ...)
        fig = prior_sensitivity_plot(
            [m1.posteriors, m2.posteriors],
            labels=['weak prior', 'strong prior'],
        )
    """
    import scipy.stats as stats

    if param_names is None:
        all_keys = [set(p.keys()) for p in posteriors_list]
        param_names = sorted(set.intersection(*all_keys))

    n_params = len(param_names)
    if n_params == 0:
        raise ValueError("No common parameters found across posteriors.")

    ncols = min(3, n_params)
    nrows = int(np.ceil(n_params / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=param_names)

    for pi, param in enumerate(param_names):
        r = pi // ncols + 1
        c = pi % ncols + 1
        for mi, (posteriors, label) in enumerate(zip(posteriors_list, labels)):
            if param not in posteriors:
                continue
            samples = np.asarray(posteriors[param]).flatten()
            color = _COLORS[mi % len(_COLORS)]

            xs = np.linspace(samples.min(), samples.max(), 300)
            kde = stats.gaussian_kde(samples)
            fig.add_trace(go.Scatter(
                x=xs, y=kde(xs), mode="lines",
                line=dict(color=color, width=2),
                name=label, showlegend=(pi == 0),
                legendgroup=label,
            ), row=r, col=c)

            # HDI marker
            lo = np.percentile(samples, 100 * (1 - hdi_prob) / 2)
            hi = np.percentile(samples, 100 * (1 - (1 - hdi_prob) / 2))
            y_max = float(kde(xs).max())
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[y_max * 0.05, y_max * 0.05],
                mode="lines", line=dict(color=color, width=4),
                showlegend=False, legendgroup=label,
            ), row=r, col=c)

    fig.update_layout(title=title, height=300 * nrows, plot_bgcolor="white")
    return fig


# =============================================================================
# 2. LOO influence
# =============================================================================

def influence_plot(
    m=None,
    log_likelihood=None,
    x=None,
    y=None,
    good_k=None,
    title="LOO Influence: Pareto-k Diagnostics",
):
    """Scatter of Pareto-k values per observation.

    High k values (k > 0.7) indicate influential observations where the model
    is sensitive — potential outliers or model misspecification.

    Args:
        m: Fitted BF model. Used to auto-compute log_likelihood if not given.
        log_likelihood: Pre-computed log-likelihood array, shape (C, S, N) or
                        (S, N). If None, computed from m.model + m.data_on_model.
        x: Optional 1D array of x-axis values (e.g. predictor). Defaults to 1..N.
        y: Optional 1D array of observed values for hover text.
        good_k: Threshold for flagging bad k (default 0.7).
        title: Plot title.

    Returns:
        plotly Figure with horizontal reference lines at k=0.5, 0.7.
    """
    from BayesForge.Diagnostic.jax_diagnostics import (
        compute_log_likelihood, _psis_weights
    )

    if log_likelihood is None:
        if m is None:
            raise ValueError("Pass m or log_likelihood=.")
        log_likelihood = compute_log_likelihood(
            m.model, m.posteriors_by_chain, **m.data_on_model
        )

    ll_arr = np.asarray(log_likelihood)
    if ll_arr.ndim == 3:
        C, S, N = ll_arr.shape
        ll_flat = ll_arr.reshape(C * S, N)
    else:
        ll_flat, N = ll_arr, ll_arr.shape[1]

    k_vals = np.array([_psis_weights(ll_flat[:, i])[1] for i in range(N)])

    if x is None:
        x = np.arange(N)
    x = np.asarray(x).flatten()
    if good_k is None:
        total_S = ll_flat.shape[0]
        good_k = min(1 - 1 / max(float(np.log10(total_S)), 1.01), 0.7)

    colors = np.where(k_vals > good_k, "red", "cornflowerblue")
    hover = [f"obs {i}: k={k_vals[i]:.3f}" +
             (f", y={float(y[i]):.3f}" if y is not None else "")
             for i in range(N)]

    fig = go.Figure()
    for clr, label in [("cornflowerblue", f"k ≤ {good_k:.2f} (good)"),
                        ("red",            f"k > {good_k:.2f} (influential)")]:
        mask = (colors == clr)
        if mask.any():
            fig.add_trace(go.Scatter(
                x=x[mask], y=k_vals[mask], mode="markers",
                marker=dict(color=clr, size=6, opacity=0.8),
                text=[hover[i] for i in np.where(mask)[0]],
                hoverinfo="text", name=label,
            ))

    fig.add_hline(y=0.5, line_dash="dash", line_color="orange",
                  annotation_text="k=0.5", annotation_position="right")
    fig.add_hline(y=good_k, line_dash="dash", line_color="red",
                  annotation_text=f"k={good_k:.2f}", annotation_position="right")
    fig.update_layout(
        title=title, xaxis_title="Observation index",
        yaxis_title="Pareto k", plot_bgcolor="white",
    )
    return fig


# =============================================================================
# 3. Calibration / coverage
# =============================================================================

def calibration_plot(
    y,
    yrep,
    levels=None,
    title="Calibration: Empirical vs Nominal Coverage",
):
    """Empirical coverage vs nominal credible interval levels.

    For each level α, computes the fraction of observed y values that fall
    inside the α-level predictive interval. A well-calibrated model has
    empirical ≈ nominal.

    Args:
        y: Observed data, shape (n_obs,).
        yrep: Posterior predictive matrix, shape (n_rep, n_obs).
        levels: List of nominal coverage levels. Defaults to
                [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99].
        title: Plot title.

    Returns:
        plotly Figure.
    """
    if levels is None:
        levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    y, = [np.asarray(y, dtype=float).flatten()]
    yrep = np.asarray(yrep, dtype=float)
    n_obs = len(y)

    empirical = []
    for level in levels:
        tail = (1 - level) / 2
        lo = np.percentile(yrep, 100 * tail, axis=0)
        hi = np.percentile(yrep, 100 * (1 - tail), axis=0)
        coverage = float(np.mean((y >= lo) & (y <= hi)))
        empirical.append(coverage)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="black", dash="dash"),
        name="Perfect calibration",
    ))
    fig.add_trace(go.Scatter(
        x=levels, y=empirical, mode="lines+markers",
        line=dict(color="cornflowerblue", width=2.5),
        marker=dict(size=8),
        name="Observed coverage",
        text=[f"{l:.0%} → {e:.1%}" for l, e in zip(levels, empirical)],
        hoverinfo="text",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Nominal coverage", yaxis_title="Empirical coverage",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        plot_bgcolor="white",
    )
    return fig


# =============================================================================
# 4. HMC diagnostics: energy + divergences
# =============================================================================

def divergence_energy_plot(m, title="HMC Diagnostics: Energy & Divergences"):
    """Plot HMC energy transition histogram and divergence locations.

    Shows the E-BFMI energy statistic (marginal energy vs transition energy)
    to check for concentration problems, and flags divergent transitions.

    Args:
        m: Fitted BF model with NumPyro MCMC sampler.
        title: Plot title.

    Returns:
        plotly Figure (two panels: energy histogram, divergences per chain).
    """
    extra = m.sampler.get_extra_fields(group_by_chain=True)
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Energy: Marginal vs Transition",
                                        "Divergences per chain"])

    # Energy panel
    if "energy" in extra and extra["energy"] is not None:
        energy = np.array(extra["energy"])
        if energy.ndim == 1:
            energy = energy[None, :]
        for c in range(energy.shape[0]):
            e = energy[c]
            color = _COLORS[c % len(_COLORS)]
            delta_e = np.diff(e)
            # marginal: histogram of e
            fig.add_trace(go.Histogram(
                x=e, name=f"Chain {c} E",
                marker_color=color, opacity=0.5,
                legendgroup=f"c{c}", showlegend=True,
            ), row=1, col=1)
            # transition: histogram of |delta_e|
            fig.add_trace(go.Histogram(
                x=np.abs(delta_e), name=f"Chain {c} |ΔE|",
                marker_color=color, opacity=0.3,
                legendgroup=f"c{c}", showlegend=False,
            ), row=1, col=1)
    else:
        fig.add_annotation(text="Energy data not available",
                           row=1, col=1, showarrow=False)

    # Divergences panel
    if "diverging" in extra and extra["diverging"] is not None:
        div = np.array(extra["diverging"])
        if div.ndim == 1:
            div = div[None, :]
        n_div_per_chain = np.sum(div, axis=1)
        total_per_chain = div.shape[1]
        chain_labels = [f"Chain {c}" for c in range(div.shape[0])]
        colors_div = [_COLORS[c % len(_COLORS)] for c in range(div.shape[0])]
        fig.add_trace(go.Bar(
            x=chain_labels, y=n_div_per_chain.tolist(),
            marker_color=colors_div, showlegend=False,
            text=[f"{d}/{total_per_chain}" for d in n_div_per_chain],
            textposition="auto",
        ), row=1, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="grey", row=1, col=2)
    else:
        fig.add_annotation(text="Divergence data not available",
                           row=1, col=2, showarrow=False)

    fig.update_layout(title=title, barmode="overlay", plot_bgcolor="white")
    return fig


# =============================================================================
# 5. Multimodality detection
# =============================================================================

def multimodality_check(
    m=None,
    posteriors=None,
    param_names=None,
    n_modes=2,
    title="Multimodality Check",
):
    """Detect multiple modes in posterior marginals using KDE + local maxima.

    Flags parameters whose KDE has more than one local maximum (potential
    multimodality). Also overlays chain-separated density to show if chains
    explore different regions.

    Args:
        m: Fitted BF model. Uses m.posteriors and m.posteriors_by_chain.
        posteriors: Dict {param: array(S,)} — override m.posteriors.
        param_names: Subset of parameters to check.
        n_modes: Number of modes to highlight.
        title: Plot title.

    Returns:
        plotly Figure, one subplot per parameter.
    """
    import scipy.stats as stats
    from scipy.signal import find_peaks

    if m is not None:
        if posteriors is None:
            posteriors = m.posteriors
        by_chain = getattr(m, 'posteriors_by_chain', None)
    else:
        by_chain = None

    if param_names is None:
        param_names = list(posteriors.keys())

    ncols = min(3, len(param_names))
    nrows = int(np.ceil(len(param_names) / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=param_names)

    for pi, param in enumerate(param_names):
        r = pi // ncols + 1
        c = pi % ncols + 1

        samples = np.asarray(posteriors[param]).flatten()
        xs = np.linspace(samples.min(), samples.max(), 500)
        kde = stats.gaussian_kde(samples)
        ys = kde(xs)

        # find modes
        peaks, _ = find_peaks(ys, prominence=ys.max() * 0.1)

        # overall density
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", fill="tozeroy",
            fillcolor="rgba(100,149,237,0.2)",
            line=dict(color="cornflowerblue", width=2),
            name=param, showlegend=False,
        ), row=r, col=c)

        # mark peaks
        for pk in peaks[:n_modes]:
            fig.add_trace(go.Scatter(
                x=[xs[pk]], y=[ys[pk]], mode="markers",
                marker=dict(color="red", size=10, symbol="triangle-up"),
                showlegend=False,
            ), row=r, col=c)

        # chains overlay if available
        if by_chain is not None and param in by_chain:
            chain_samples = np.asarray(by_chain[param])
            for ci in range(chain_samples.shape[0]):
                cs = chain_samples[ci].flatten()
                kde_c = stats.gaussian_kde(cs)
                color = _COLORS[ci % len(_COLORS)].replace("#", "")
                fig.add_trace(go.Scatter(
                    x=xs, y=kde_c(xs), mode="lines",
                    line=dict(color=_COLORS[ci % len(_COLORS)], width=1,
                               dash="dot"),
                    name=f"Chain {ci}", showlegend=(pi == 0),
                    legendgroup=f"chain{ci}",
                ), row=r, col=c)

        n_peaks = len(peaks)
        flag = " ⚠ multimodal" if n_peaks > 1 else " ✓"
        fig.update_annotations(
            selector=dict(text=param),
            text=param + flag,
        )

    fig.update_layout(title=title, height=300 * nrows, plot_bgcolor="white")
    return fig
