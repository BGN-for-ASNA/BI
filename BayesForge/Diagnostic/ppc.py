"""Posterior Predictive Check (PPC) plots for BF models.

All PPC functions take:
    y    : 1D array of observed values, shape (n_obs,)
    yrep : 2D matrix of replicated datasets, shape (n_rep, n_obs)

Generate yrep with:
    pred = m.sample()
    yrep = np.asarray(pred[m.obs_args[0]])   # (S, n_obs)

or use the helper:
    yrep = get_yrep(m)

Categories implemented
-----------------------
Distributions  : ppc_density, ppc_hist, ppc_boxplot
Statistics     : ppc_stat, ppc_stat_2d
Intervals      : ppc_intervals, ppc_ribbon
Errors         : ppc_error_scatter, ppc_error_hist
Scatterplots   : ppc_scatter
Discrete       : ppc_rootogram, ppc_bars
LOO            : ppc_loo_pit, ppc_loo_intervals
"""
import numpy as np
import jax.numpy as jnp
import plotly.graph_objects as go
import plotly.colors as pcolors
from plotly.subplots import make_subplots

_COLORS = pcolors.qualitative.Plotly
_REP_COLOR = "rgba(100,149,237,0.25)"
_REP_COLOR_SOLID = "rgba(100,149,237,0.6)"
_OBS_COLOR = "black"


# =============================================================================
# Utility
# =============================================================================

def get_yrep(m, seed=0):
    """Return posterior predictive matrix yrep of shape (S, n_obs).

    Calls m.sample() with the training predictors (obs removed), collects
    the observed-variable site from all posterior draws.
    """
    import jax
    pred_data = {k: v for k, v in m.data_on_model.items()
                 if k not in (m.obs_args or [])}
    pred = m.sample(data=pred_data, remove_obs=False, posterior=True, seed=seed)
    param_keys = set(m.posteriors.keys()) if hasattr(m, 'posteriors') and m.posteriors else set()
    if m.obs_args and m.obs_args[0] in pred:
        obs_key = m.obs_args[0]
    else:
        # Picking "the first non-parameter key" depends on dict ordering and
        # silently returns the wrong site for models with deterministic sites.
        candidates = [k for k in pred if k not in param_keys]
        if len(candidates) != 1:
            raise ValueError(
                f"Cannot identify the observed site: m.obs_args={m.obs_args!r} "
                f"and the predictive returned {candidates!r}. Pass yrep= "
                "explicitly, or set m.obs_args."
            )
        obs_key = candidates[0]
    yrep = np.asarray(pred[obs_key])
    if yrep.ndim > 2:
        yrep = yrep.reshape(yrep.shape[0], -1)
    return yrep


def _as_np(*arrays):
    return [np.asarray(a, dtype=float).flatten() for a in arrays]


def _kde(stats, data, bw_adjust=1.0):
    """gaussian_kde with bw_adjust as a MULTIPLIER on Scott's rule.

    scipy's ``bw_method=`` is the absolute bandwidth factor, not an adjustment,
    so passing bw_adjust straight through made the default (1.0) about 3.5x
    wider than Scott at n=500 -- every density overlay came out oversmoothed.
    """
    kde = stats.gaussian_kde(data)
    if bw_adjust is not None and bw_adjust != 1.0:
        kde.set_bandwidth(kde.factor * float(bw_adjust))
    return kde


def _stat_fn(name):
    fns = {
        "mean":   np.mean,
        "median": np.median,
        "sd":     lambda x: float(np.std(x, ddof=1)),
        "var":    lambda x: float(np.var(x, ddof=1)),
        "min":    np.min,
        "max":    np.max,
        "q25":    lambda x: np.percentile(x, 25),
        "q75":    lambda x: np.percentile(x, 75),
        "skew":   lambda x: float(np.mean(((x - np.mean(x)) / (np.std(x) + 1e-30)) ** 3)),
        "n_zero": lambda x: int(np.sum(x == 0)),
        "n_pos":  lambda x: int(np.sum(x > 0)),
    }
    if callable(name):
        return name
    if name not in fns:
        raise ValueError(f"Unknown stat '{name}'. Choose from {list(fns)}")
    return fns[name]


# =============================================================================
# Distributions
# =============================================================================

def ppc_density(y, yrep, n=50, bw_adjust=1.0, title="PPC: Density overlay"):
    """KDE of y (black) vs n random draws from yrep rows (blue).

    Args:
        y: Observed data, shape (n_obs,).
        yrep: Posterior predictive matrix, shape (n_rep, n_obs).
        n: Max number of yrep lines to show.
        bw_adjust: Bandwidth multiplier passed to scipy KDE.
        title: Plot title.
    Returns:
        plotly Figure.
    """
    import scipy.stats as stats
    y, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)

    finite_mask = np.isfinite(yrep).all(axis=1)
    yrep_clean = yrep[finite_mask]
    if len(yrep_clean) == 0:
        raise ValueError("yrep contains no finite rows.")
    n_dropped = len(yrep) - len(yrep_clean)
    if n_dropped:
        import warnings
        warnings.warn(
            f"ppc_density dropped {n_dropped} of {len(yrep)} yrep draws "
            "containing non-finite values.", stacklevel=2)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(yrep_clean), size=min(n, len(yrep_clean)), replace=False)

    p1    = float(np.percentile(y, 0.5))
    p99_y = float(np.percentile(y, 99.5))
    p1_r  = float(np.percentile(yrep_clean[idx], 0.5))
    p99_r = float(np.percentile(yrep_clean[idx], 99.5))
    x_min = float(min(p1, p1_r))
    x_max = float(max(p99_y, p99_r))
    if x_min >= x_max:
        x_max = x_min + 1.0
    xs = np.linspace(x_min, x_max, 300, dtype=np.float64)

    fig = go.Figure()
    for i, s in enumerate(idx):
        row = np.asarray(yrep_clean[s], dtype=np.float64)
        row = row[np.isfinite(row)]
        if len(row) < 2 or float(np.std(row)) < 1e-12:
            continue
        try:
            kde = _kde(stats, row, bw_adjust)
            vals = kde(xs)
            if not np.isfinite(vals).all():
                continue
        except Exception:
            continue
        fig.add_trace(go.Scatter(
            x=xs, y=vals, mode="lines",
            line=dict(color=_REP_COLOR, width=1),
            showlegend=(i == 0), name="yrep",
        ))

    y_clean = np.asarray(y, dtype=np.float64).flatten()
    y_clean = y_clean[np.isfinite(y_clean)]
    if len(y_clean) >= 2 and float(np.std(y_clean)) > 1e-12:
        kde_y = _kde(stats, y_clean, bw_adjust)
        fig.add_trace(go.Scatter(
            x=xs, y=kde_y(xs), mode="lines",
            line=dict(color=_OBS_COLOR, width=2.5),
            name="y (observed)",
        ))

    fig.update_layout(
        title=title, xaxis_title="Value", yaxis_title="Density",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def ppc_hist(y, yrep, n=8, bins=30, title="PPC: Histogram small multiples"):
    """Histograms: observed y plus n random yrep draws.

    Args:
        y: Observed data.
        yrep: Posterior predictive matrix.
        n: Number of yrep draws to show.
        bins: Number of histogram bins.
        title: Plot title.
    Returns:
        plotly Figure.
    """
    y, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(yrep), size=min(n, len(yrep)), replace=False)

    total = 1 + len(idx)
    ncols = min(4, total)
    nrows = int(np.ceil(total / ncols))
    subtitles = ["y (observed)"] + [f"yrep[{s}]" for s in idx]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subtitles)

    def _add_hist(data, row, col, color):
        fig.add_trace(go.Histogram(
            x=data, nbinsx=bins,
            marker_color=color, opacity=0.75, showlegend=False,
        ), row=row, col=col)

    _add_hist(y, 1, 1, _OBS_COLOR)
    for i, s in enumerate(idx):
        r = (i + 1) // ncols + 1
        c = (i + 1) % ncols + 1
        _add_hist(yrep[s], r, c, "cornflowerblue")

    fig.update_layout(title=title, height=280 * nrows, showlegend=False,
                      plot_bgcolor="white")
    return fig


def ppc_boxplot(y, yrep, n=20, title="PPC: Boxplots"):
    """Box-and-whisker: y (black) vs n yrep draws.

    Args:
        y: Observed data.
        yrep: Posterior predictive matrix.
        n: Number of yrep draws.
        title: Plot title.
    Returns:
        plotly Figure.
    """
    y, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(yrep), size=min(n, len(yrep)), replace=False)

    fig = go.Figure()
    for s in idx:
        fig.add_trace(go.Box(y=yrep[s], marker_color=_REP_COLOR_SOLID,
                             showlegend=False, name=f"rep {s}"))
    fig.add_trace(go.Box(y=y, marker_color=_OBS_COLOR,
                         name="y (observed)", showlegend=True))
    fig.update_layout(title=title, yaxis_title="Value",
                      plot_bgcolor="white", boxmode="overlay")
    return fig


# =============================================================================
# Test statistics
# =============================================================================

def ppc_stat(y, yrep, stat="mean", title=None):
    """Distribution of test statistic over yrep vs observed value.

    Args:
        y: Observed data.
        yrep: Posterior predictive matrix.
        stat: Statistic name or callable. Options: 'mean', 'median', 'sd',
              'var', 'min', 'max', 'q25', 'q75', 'skew', 'n_zero', 'n_pos'.
        title: Plot title.
    Returns:
        plotly Figure with Bayesian p-value annotation.
    """
    import scipy.stats as scipy_stats
    y, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)
    fn = _stat_fn(stat)
    stat_name = stat if isinstance(stat, str) else "T"

    t_y = float(fn(y))
    t_rep = np.array([fn(yrep[s]) for s in range(len(yrep))])

    # One-sided P(T(yrep) >= T(y)); the two-sided value is what people usually
    # mean by "the" Bayesian p-value, so report both rather than an
    # unqualified number that reads as 0 or 1 for a good fit.
    pval = float(np.mean(t_rep >= t_y))
    pval_2s = float(2 * min(pval, 1.0 - pval))

    xs = np.linspace(t_rep.min(), t_rep.max(), 300)
    kde = _kde(scipy_stats, t_rep)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=kde(xs), mode="lines", fill="tozeroy",
        fillcolor="rgba(100,149,237,0.3)",
        line=dict(color="cornflowerblue", width=1.5),
        name=f"T(yrep) = {stat_name}",
    ))
    fig.add_vline(x=t_y, line_color=_OBS_COLOR, line_width=2.5,
                  annotation_text=f"T(y) = {t_y:.3g}", annotation_position="top right")
    fig.update_layout(
        title=title or (f"PPC: {stat_name}  |  P(T(yrep) ≥ T(y)) = {pval:.3f}"
                        f"  (two-sided p = {pval_2s:.3f})"),
        xaxis_title=f"T = {stat_name}(·)",
        yaxis_title="Density",
        plot_bgcolor="white",
    )
    return fig


def ppc_stat_2d(y, yrep, stat1="mean", stat2="sd", title=None):
    """Scatter of two test statistics over yrep vs observed point.

    Args:
        y: Observed data.
        yrep: Posterior predictive matrix.
        stat1, stat2: Statistic names or callables.
        title: Plot title.
    Returns:
        plotly Figure.
    """
    y, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)
    fn1, fn2 = _stat_fn(stat1), _stat_fn(stat2)
    s1_name = stat1 if isinstance(stat1, str) else "T1"
    s2_name = stat2 if isinstance(stat2, str) else "T2"

    t1_rep = np.array([fn1(yrep[s]) for s in range(len(yrep))])
    t2_rep = np.array([fn2(yrep[s]) for s in range(len(yrep))])
    t1_y, t2_y = float(fn1(y)), float(fn2(y))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t1_rep, y=t2_rep, mode="markers",
        marker=dict(color="cornflowerblue", opacity=0.4, size=5),
        name="yrep",
    ))
    fig.add_trace(go.Scatter(
        x=[t1_y], y=[t2_y], mode="markers",
        marker=dict(color=_OBS_COLOR, size=12, symbol="x"),
        name="y (observed)",
    ))
    fig.update_layout(
        title=title or f"PPC: {s1_name} vs {s2_name}",
        xaxis_title=s1_name, yaxis_title=s2_name,
        plot_bgcolor="white",
    )
    return fig


# =============================================================================
# Intervals
# =============================================================================

def ppc_intervals(y, yrep, x=None, prob=0.5, prob_outer=0.9,
                  title="PPC: Predictive Intervals"):
    """Interval estimates of yrep with y overlaid.

    Plots inner (prob) and outer (prob_outer) credible intervals computed
    from the columns of yrep, with observed y as scatter.

    Args:
        y: Observed data, shape (n_obs,).
        yrep: Posterior predictive matrix, shape (n_rep, n_obs).
        x: Optional x-axis variable, shape (n_obs,). Defaults to 1..n_obs.
        prob: Inner interval probability (e.g., 0.5).
        prob_outer: Outer interval probability (e.g., 0.9).
        title: Plot title.
    Returns:
        plotly Figure.
    """
    y, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)
    x_was_none = x is None
    if x is None:
        x = np.arange(len(y))
    x = np.asarray(x).flatten()

    # sort by x for cleaner ribbon
    order = np.argsort(x)
    x = x[order]; y = y[order]; yrep = yrep[:, order]

    lo_out = np.percentile(yrep, 100 * (1 - prob_outer) / 2, axis=0)
    hi_out = np.percentile(yrep, 100 * (1 - (1 - prob_outer) / 2), axis=0)
    lo_in  = np.percentile(yrep, 100 * (1 - prob) / 2, axis=0)
    hi_in  = np.percentile(yrep, 100 * (1 - (1 - prob) / 2), axis=0)
    med    = np.median(yrep, axis=0)

    fig = go.Figure()
    # outer ribbon
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([hi_out, lo_out[::-1]]),
        fill="toself", fillcolor="rgba(100,149,237,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"{int(prob_outer*100)}% interval",
    ))
    # inner ribbon
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([hi_in, lo_in[::-1]]),
        fill="toself", fillcolor="rgba(100,149,237,0.35)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"{int(prob*100)}% interval",
    ))
    # median
    fig.add_trace(go.Scatter(
        x=x, y=med, mode="lines",
        line=dict(color="cornflowerblue", width=1.5),
        name="Median yrep",
    ))
    # observed
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(color=_OBS_COLOR, size=5, opacity=0.8),
        name="y (observed)",
    ))
    fig.update_layout(title=title, xaxis_title="Index" if x_was_none else "x",
                      yaxis_title="Value", plot_bgcolor="white")
    return fig


def ppc_ribbon(y, yrep, x=None, prob=0.5, prob_outer=0.9,
               title="PPC: Predictive Ribbon"):
    """Continuous ribbon variant of ppc_intervals. See ppc_intervals."""
    return ppc_intervals(y, yrep, x=x, prob=prob, prob_outer=prob_outer,
                         title=title)


# =============================================================================
# Errors
# =============================================================================

def ppc_error_scatter(y, yrep, title="PPC: y vs E[yrep]"):
    """Scatter of observed y vs posterior predictive mean E[yrep].

    Perfect model: points fall on y = x diagonal.

    Args:
        y: Observed data.
        yrep: Posterior predictive matrix.
        title: Plot title.
    Returns:
        plotly Figure.
    """
    y, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)
    mu_rep = yrep.mean(axis=0)

    lim = [min(y.min(), mu_rep.min()), max(y.max(), mu_rep.max())]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mu_rep, y=y, mode="markers",
        marker=dict(color="cornflowerblue", size=6, opacity=0.7),
        name="observations",
    ))
    fig.add_trace(go.Scatter(
        x=lim, y=lim, mode="lines",
        line=dict(color="black", width=1.5, dash="dash"),
        name="y = x",
    ))
    fig.update_layout(title=title, xaxis_title="E[yrep]", yaxis_title="y",
                      plot_bgcolor="white")
    return fig


def ppc_error_hist(y, yrep, bins=40, title="PPC: Posterior residuals"):
    """Distribution of mean residuals r_i = y_i - E[yrep_i].

    Args:
        y: Observed data.
        yrep: Posterior predictive matrix.
        bins: Number of histogram bins.
        title: Plot title.
    Returns:
        plotly Figure.
    """
    y, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)
    residuals = y - yrep.mean(axis=0)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=residuals, nbinsx=bins,
                               marker_color="cornflowerblue", opacity=0.8,
                               showlegend=False))
    fig.add_vline(x=0, line_color="black", line_dash="dash")
    fig.update_layout(title=title, xaxis_title="y - E[yrep]",
                      yaxis_title="Count", plot_bgcolor="white")
    return fig


# =============================================================================
# Scatterplots
# =============================================================================

def ppc_scatter(y, yrep, n_reps=9, title="PPC: y vs individual yrep"):
    """Small-multiple scatterplots of y vs n_reps random rows of yrep.

    Args:
        y: Observed data.
        yrep: Posterior predictive matrix.
        n_reps: Number of replications to show.
        title: Plot title.
    Returns:
        plotly Figure.
    """
    y, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(yrep), size=min(n_reps, len(yrep)), replace=False)

    ncols = 3
    nrows = int(np.ceil(len(idx) / ncols))
    subtitles = [f"yrep[{s}]" for s in idx]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subtitles)

    lim = [y.min(), y.max()]
    for i, s in enumerate(idx):
        r = i // ncols + 1
        c = i % ncols + 1
        fig.add_trace(go.Scatter(
            x=yrep[s], y=y, mode="markers",
            marker=dict(color="cornflowerblue", size=5, opacity=0.6),
            showlegend=False,
        ), row=r, col=c)
        fig.add_trace(go.Scatter(
            x=lim, y=lim, mode="lines",
            line=dict(color="black", dash="dash", width=1),
            showlegend=False,
        ), row=r, col=c)
        fig.update_xaxes(title_text="yrep", row=r, col=c)
        fig.update_yaxes(title_text="y", row=r, col=c)

    fig.update_layout(title=title, height=300 * nrows, plot_bgcolor="white")
    return fig


# =============================================================================
# Discrete
# =============================================================================

def ppc_rootogram(y, yrep, title="PPC: Rootogram (count data)"):
    """Hanging rootogram for count outcomes.

    Compares sqrt of observed counts (bars) to expected sqrt counts (curve).
    Bars hanging below zero indicate over-prediction; above zero: under-prediction.

    Args:
        y: Observed integer counts.
        yrep: Posterior predictive matrix of counts.
        title: Plot title.
    Returns:
        plotly Figure.
    """
    y = np.rint(np.asarray(y, dtype=float).flatten()).astype(int)
    yrep = np.rint(np.asarray(yrep, dtype=float)).astype(int)
    if y.min() < 0 or yrep.min() < 0:
        raise ValueError("ppc_rootogram expects non-negative counts.")

    max_val = max(int(y.max()), int(yrep.max()))
    counts_obs = np.bincount(y, minlength=max_val + 1).astype(float)

    counts_rep = np.array([
        np.bincount(yrep[s], minlength=max_val + 1)
        for s in range(len(yrep))
    ], dtype=float)
    expected = counts_rep.mean(axis=0)

    xs = np.arange(len(counts_obs))
    sqrt_obs = np.sqrt(counts_obs)
    sqrt_exp = np.sqrt(expected)

    fig = go.Figure()
    # Hanging rootogram: bars of height sqrt(observed) are suspended FROM the
    # sqrt(expected) curve, so their distance from y=0 is the misfit and the
    # zero line is the reference. Basing them at sqrt_exp instead (as before)
    # drew bars spanning expected->observed, leaving add_hline(y=0) meaningless.
    fig.add_trace(go.Bar(
        x=xs, y=sqrt_obs, base=sqrt_exp - sqrt_obs,
        marker_color="cornflowerblue", opacity=0.7,
        name="√Observed (hanging from √Expected)",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=sqrt_exp, mode="lines+markers",
        line=dict(color="black", width=2),
        name="E[√yrep]",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_layout(
        title=title, xaxis_title="Count value",
        yaxis_title="√Frequency",
        plot_bgcolor="white",
    )
    return fig


def ppc_bars(y, yrep, title="PPC: Bar chart (discrete/categorical)"):
    """Bar chart comparing observed y to posterior predictive yrep.

    For ordinal, categorical, or discrete outcomes.
    Shows observed counts and yrep credible interval per category.

    Args:
        y: Observed discrete values.
        yrep: Posterior predictive matrix.
        title: Plot title.
    Returns:
        plotly Figure.
    """
    y = np.asarray(y).flatten()
    yrep = np.asarray(yrep)
    cats = np.unique(np.concatenate([y, yrep.flatten()]))
    if len(cats) > 50:
        raise ValueError(
            f"ppc_bars found {len(cats)} distinct values; it is meant for "
            "discrete/categorical outcomes. Use ppc_density or ppc_hist for "
            "continuous data."
        )

    obs_counts = np.array([np.sum(y == c) for c in cats], dtype=float)
    rep_counts = np.array([
        [np.sum(yrep[s] == c) for c in cats]
        for s in range(len(yrep))
    ], dtype=float)

    rep_mean = rep_counts.mean(axis=0)
    rep_lo   = np.percentile(rep_counts, 5, axis=0)
    rep_hi   = np.percentile(rep_counts, 95, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cats, y=obs_counts,
        marker_color=_OBS_COLOR, opacity=0.8,
        name="y (observed)",
    ))
    fig.add_trace(go.Scatter(
        x=cats, y=rep_mean, mode="markers",
        marker=dict(color="cornflowerblue", size=10, symbol="diamond"),
        error_y=dict(type="data", symmetric=False,
                     array=rep_hi - rep_mean,
                     arrayminus=rep_mean - rep_lo),
        name="E[yrep] ± 90% CI",
    ))
    fig.update_layout(title=title, xaxis_title="Category",
                      yaxis_title="Count", plot_bgcolor="white")
    return fig


# =============================================================================
# LOO-based PPC
# =============================================================================

def ppc_loo_pit(y, yrep, log_likelihood=None, reff=None,
                title="PPC: LOO-PIT (uniformity check)"):
    """LOO probability integral transform check.

    PIT value for each observation: ``F(y_i) = P(yrep[:, i] <= y_i)``.
    Under a well-calibrated model these are Uniform(0, 1).

    When ``log_likelihood`` is given, the draws are reweighted by
    Pareto-smoothed importance sampling so each ``F(y_i)`` is a *leave-one-out*
    predictive CDF — this is what makes the check meaningful. Without it the
    same draws that were fitted to ``y`` are used to score ``y``, which pulls
    the PITs toward 0.5 and makes a miscalibrated model look calibrated; a
    warning is emitted in that case.

    Args:
        y: Observed data, shape (n_obs,).
        yrep: Posterior predictive matrix, shape (n_rep, n_obs).
        log_likelihood: Pointwise log-likelihood, shape (C, S, n_obs) or
            (S, n_obs), with S matching yrep's rows. Enables true LOO-PIT.
        reff: Relative MCMC efficiency for PSIS; see
            ``jax_diagnostics.relative_eff``. Defaults to 1.0.
        title: Plot title.
    Returns:
        plotly Figure.
    """
    import warnings
    import scipy.stats as stats
    y, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)
    n_obs = len(y)

    if log_likelihood is None:
        warnings.warn(
            "ppc_loo_pit called without log_likelihood: computing a plain "
            "posterior-predictive PIT, not LOO-PIT. The same draws are used to "
            "fit and to score y, so the PITs are biased toward 0.5 and a "
            "miscalibrated model can look calibrated. Pass log_likelihood= for "
            "the leave-one-out version.",
            stacklevel=2,
        )
        pit = np.array([np.mean(yrep[:, i] <= y[i]) for i in range(n_obs)],
                       dtype=np.float64)
    else:
        from BayesForge.Diagnostic.jax_diagnostics import _psis_weights
        ll = np.asarray(log_likelihood, dtype=np.float64)
        if ll.ndim == 3:
            ll = ll.reshape(ll.shape[0] * ll.shape[1], ll.shape[2])
        if ll.shape != yrep.shape:
            raise ValueError(
                f"log_likelihood shape {ll.shape} does not match yrep "
                f"{yrep.shape}; both need (n_draws, n_obs) with the same draws."
            )
        pit = np.empty(n_obs, dtype=np.float64)
        for i in range(n_obs):
            log_w, _ = _psis_weights(ll[:, i], reff=1.0 if reff is None else reff)
            w = np.exp(np.asarray(log_w, dtype=np.float64))
            w /= w.sum()
            pit[i] = float(np.sum(w * (yrep[:, i] <= y[i])))

    pit = pit[np.isfinite(pit)]
    if len(pit) < 2:
        import plotly.graph_objects as go2
        return go2.Figure().update_layout(title=title + " (insufficient data)")

    xs = np.linspace(0, 1, 300)
    kde = _kde(stats, pit)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=kde(xs), mode="lines", fill="tozeroy",
        fillcolor="rgba(100,149,237,0.3)",
        line=dict(color="cornflowerblue", width=2),
        name="PIT KDE",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[1, 1], mode="lines",
        line=dict(color="black", dash="dash", width=1.5),
        name="Uniform(0,1)",
    ))
    fig.update_layout(
        title=title, xaxis_title="PIT", yaxis_title="Density",
        xaxis=dict(range=[0, 1]),
        plot_bgcolor="white",
    )
    return fig


def ppc_intervals_sorted(y, yrep, x=None, prob=0.9,
                         title="PPC: Predictive intervals (sorted by y)"):
    """Predictive intervals ordered by the observed value.

    This is ``ppc_intervals`` with the observations sorted by y, which makes
    systematic over/under-prediction across the outcome range easy to read.

    NOTE: it performs no leave-one-out computation. It was previously named
    ``ppc_loo_intervals``, which promised LOO predictive intervals it never
    computed; ``ppc_loo_intervals`` remains as a deprecated alias.

    Args:
        y: Observed data.
        yrep: Posterior predictive matrix.
        x: Optional x-axis ordering variable.
        prob: Interval probability.
        title: Plot title.
    Returns:
        plotly Figure.
    """
    if x is None:
        x = np.argsort(np.asarray(y, dtype=float).flatten())
    return ppc_intervals(y, yrep, x=x, prob=prob * 0.5, prob_outer=prob,
                         title=title)


def ppc_loo_intervals(y, yrep, x=None, prob=0.9,
                      title="PPC: Predictive intervals (sorted by y)"):
    """Deprecated alias for :func:`ppc_intervals_sorted` (it is not LOO-based)."""
    import warnings
    warnings.warn(
        "ppc_loo_intervals performs no leave-one-out computation; it is "
        "ppc_intervals sorted by y. Use ppc_intervals_sorted instead.",
        DeprecationWarning, stacklevel=2,
    )
    return ppc_intervals_sorted(y, yrep, x=x, prob=prob, title=title)
