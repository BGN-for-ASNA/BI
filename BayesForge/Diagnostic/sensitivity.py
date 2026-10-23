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
from BayesForge.Diagnostic.ppc import _axis_labels
import plotly.colors as pcolors
from plotly.subplots import make_subplots
import plotly.express as px

_COLORS = pcolors.qualitative.Plotly


def _hdi(samples, prob):
    """Highest-density interval (lower, upper) of a 1D sample."""
    from BayesForge.Diagnostic.jax_diagnostics import hdi as _jd_hdi
    bounds = np.asarray(_jd_hdi(np.asarray(samples).flatten(), hdi_prob=prob))
    return float(bounds[0]), float(bounds[1])


# =============================================================================
# 1. Prior sensitivity
# =============================================================================

def prior_sensitivity_plot(
    posteriors_list,
    labels,
    param_names=None,
    hdi_prob=0.89,
    title="Prior Sensitivity Analysis",
    x_label=None,
    y_label=None,
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
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.

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

            # HDI marker -- the actual highest-density interval, not an
            # equal-tailed percentile interval. The two differ for the skewed
            # posteriors this plot exists to compare.
            lo, hi = _hdi(samples, hdi_prob)
            y_max = float(kde(xs).max())
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[y_max * 0.05, y_max * 0.05],
                mode="lines", line=dict(color=color, width=4),
                showlegend=False, legendgroup=label,
            ), row=r, col=c)

    fig.update_layout(title=title, height=300 * nrows, plot_bgcolor="white")
    return _axis_labels(fig, x_label, y_label)


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
    x_label=None,
    y_label=None,
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
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.

    Returns:
        plotly Figure with horizontal reference lines at k=0.5, 0.7.
    """
    from BayesForge.Diagnostic.jax_diagnostics import (
        compute_log_likelihood, _psis_weights, relative_eff
    )

    reff = 1.0
    if log_likelihood is None:
        if m is None:
            raise ValueError("Pass m or log_likelihood=.")
        log_likelihood = compute_log_likelihood(
            m.model, m.posteriors_by_chain, **m.data_on_model
        )
    if m is not None and getattr(m, "posteriors_by_chain", None):
        # PSIS sizes its Pareto tail as 3*sqrt(S/reff); leaving reff at 1 for
        # autocorrelated draws biases exactly the k values this plot thresholds.
        reff = relative_eff(m.posteriors_by_chain)

    ll_arr = np.asarray(log_likelihood)
    if ll_arr.ndim == 3:
        C, S, N = ll_arr.shape
        ll_flat = ll_arr.reshape(C * S, N)
    else:
        ll_flat, N = ll_arr, ll_arr.shape[1]

    k_vals = np.array([_psis_weights(ll_flat[:, i], reff=reff)[1] for i in range(N)])

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
    return _axis_labels(fig, x_label, y_label)


# =============================================================================
# 3. Calibration / coverage
# =============================================================================

def calibration_plot(
    y,
    yrep,
    levels=None,
    title=None,
    name="y",
    x_label=None,
    y_label=None,
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
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.

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
        title=title or f"Calibration — predictions of {name}",
        xaxis_title="Nominal level of the predictive interval",
        yaxis_title=f"Share of observed {name} inside it",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        plot_bgcolor="white",
    )
    return _axis_labels(fig, x_label, y_label)


# =============================================================================
# 4. HMC diagnostics: energy + divergences
# =============================================================================

def divergence_energy_plot(m, title="HMC Diagnostics: Energy & Divergences",
                           x_label=None, y_label=None):
    """Plot the HMC energy diagnostic and the divergent transitions per chain.

    Top row, one panel per chain (as bayesplot's ``mcmc_nuts_energy``): the
    distribution of the energy E around its mean (marginal energy) and of its
    change between successive iterations, Delta E (energy transition). When
    the Delta E histogram is about as wide as the E histogram the sampler can
    move across the whole posterior; a much narrower Delta E histogram means
    it explores the tails badly. The panel title gives E-BFMI =
    sum(Delta E^2) / sum((E - mean E)^2); values below 0.3 are flagged.

    Bottom row: number of divergent transitions per chain (ideal: 0).

    Args:
        m: Fitted BF model with NumPyro MCMC sampler.
        title: Plot title.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.

    Returns:
        plotly Figure.
    """
    # TFP / SVI samplers do not expose extra_fields; the panels below already
    # render an "unavailable" annotation, so degrade rather than raise.
    try:
        extra = m.sampler.get_extra_fields(group_by_chain=True) or {}
    except (AttributeError, TypeError):
        extra = {}

    def _per_chain(key):
        v = extra.get(key)
        if v is None:
            return None
        v = np.asarray(v)
        return v[None, :] if v.ndim == 1 else v

    energy, div = _per_chain("energy"), _per_chain("diverging")
    n_chains = next((len(v) for v in (energy, div) if v is not None), 1)

    subtitles = []
    for c in range(n_chains):
        if energy is None:
            subtitles.append(f"Chain {c}")
            continue
        e = energy[c].astype(float)
        bfmi = float(np.sum(np.diff(e) ** 2) / np.sum((e - e.mean()) ** 2))
        subtitles.append(f"Chain {c}: E-BFMI = {bfmi:.2f}" + (" ⚠" if bfmi < 0.3 else ""))
    subtitles.append("Divergent transitions per chain")
    fig = make_subplots(
        rows=2, cols=n_chains, subplot_titles=subtitles,
        specs=[[{}] * n_chains, [{"colspan": n_chains}] + [None] * (n_chains - 1)],
        vertical_spacing=0.18,
    )

    if energy is not None:
        for c in range(n_chains):
            e = energy[c].astype(float)
            for vals, label, color in [(e - e.mean(), "E − mean(E)  (marginal energy)", "cornflowerblue"),
                                       (np.diff(e), "ΔE  (energy transition)", "orange")]:
                fig.add_trace(go.Histogram(
                    x=vals, histnorm="probability density", nbinsx=40,
                    marker_color=color, opacity=0.55, name=label,
                    legendgroup=label, showlegend=(c == 0),
                ), row=1, col=c + 1)
            fig.update_xaxes(title_text="Energy", row=1, col=c + 1)
    else:
        fig.add_annotation(text="Energy data not available", showarrow=False,
                           xref="x domain", yref="y domain", x=0.5, y=0.5)

    if div is not None:
        n_div = div.sum(axis=1).astype(int)
        fig.add_trace(go.Bar(
            x=[f"Chain {c}" for c in range(n_chains)], y=n_div.tolist(),
            marker_color=["firebrick" if d else "cornflowerblue" for d in n_div],
            text=[f"{d} / {div.shape[1]}" for d in n_div], textposition="outside",
            showlegend=False,
        ), row=2, col=1)
        fig.update_yaxes(title_text="Divergences", rangemode="tozero",
                         range=[0, max(1, int(n_div.max())) * 1.25], row=2, col=1)
        if n_div.sum() == 0:
            fig.add_annotation(text="No divergent transitions", showarrow=False,
                               row=2, col=1, x=0.5, y=0.5,
                               xref="x domain", yref="y domain",
                               font=dict(color="seagreen", size=14))
    else:
        fig.add_annotation(text="Divergence data not available", showarrow=False,
                           row=2, col=1, x=0.5, y=0.5,
                           xref="x domain", yref="y domain")

    fig.update_layout(title=title, barmode="overlay", plot_bgcolor="white",
                      height=700,
                      legend=dict(orientation="h", yanchor="top", y=-0.08))
    return _axis_labels(fig, x_label, y_label)


# =============================================================================
# 5. Multimodality detection
# =============================================================================

def multimodality_check(
    m=None,
    posteriors=None,
    param_names=None,
    n_modes=2,
    title="Multimodality Check",
    x_label=None,
    y_label=None,
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
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.

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
        # Index this subplot's own annotation. update_annotations(selector=...)
        # rewrote EVERY annotation whose text matched, including the figure
        # title and any duplicate subplot title.
        if pi < len(fig.layout.annotations):
            fig.layout.annotations[pi].text = param + flag

    fig.update_layout(title=title, height=300 * nrows, plot_bgcolor="white")
    return _axis_labels(fig, x_label, y_label)
