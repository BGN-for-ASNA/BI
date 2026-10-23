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
Intervals      : ppc_intervals, ppc_ribbon, ppc_intervals_sorted
Errors         : ppc_error_scatter, ppc_error_hist
Scatterplots   : ppc_scatter
Discrete       : ppc_rootogram, ppc_bars
LOO            : ppc_loo_pit (ppc_loo_intervals: deprecated alias of ppc_intervals_sorted)
"""
import re

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

def get_yrep(m, site=None, seed=None):
    """Return posterior predictive matrix yrep of shape (S, n_obs), or a dict
    of such matrices if the model has multiple observed sites and ``site=None``.

    The observed sites are simulated from every posterior draw (their data
    entries are set to None), so yrep holds genuine replicated data -- never the
    observed values themselves.

    Args:
        m (bf): Fitted BayesForge model.
        site (str, optional): Name of the observed site. If None and the model
            has a single observed site, that site is used. If None and the model
            has multiple observed sites, a dict ``{site_name: yrep}`` is returned.
        seed (int, optional): Seed of the predictive simulation. Defaults to None.

    Returns:
        np.ndarray or dict[str, np.ndarray]: Matrix of shape (S, n_obs) or
        dict mapping each observed site name to its replicated data matrix.
    """
    sites = m._observed_sites(m.data_on_model)
    if not sites:
        raise ValueError("No observed sites found in the model.")

    if site is not None:
        if site not in sites:
            raise ValueError(
                f"Unknown observed site {site!r}. Available sites: {sorted(sites)}"
            )
        target_sites = {site: sites[site]}
    else:
        target_sites = sites

    for s_name, key in target_sites.items():
        if key is None:
            raise ValueError(
                f"The observed site {s_name!r} is not one of the model's data entries "
                "(e.g. obs=y - 1), so it cannot be simulated. Pass yrep= explicitly."
            )

    pred = m.sample(remove_obs=True, posterior=True, seed=seed)

    yreps = {}
    for s_name in target_sites:
        arr = np.asarray(pred[s_name])
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        yreps[s_name] = arr

    if site is not None or len(sites) == 1:
        return next(iter(yreps.values())) if site is None else yreps[site]
    return yreps


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


# Every plot labels the outcome by ``name`` -- m.diag passes the observed site
# name (e.g. "height") -- so axes and legends read "Predictions of height"
# rather than the bayesplot jargon "yrep".
def _obs(name):
    return f"{name} (observed)"


def _preds(name):
    return f"Predictions of {name}"


def _axis_labels(fig, x_label=None, y_label=None):
    """Override the axis titles of every plot: a string sets all the x (y)
    axes, a list sets them one per subplot in layout order (None keeps one)."""
    for label, axes in ((x_label, "xaxis"), (y_label, "yaxis")):
        if label is None:
            continue
        names = sorted((k for k in fig.layout.to_plotly_json() if re.fullmatch(axes + r"\d*", k)),
                       key=lambda k: int(k[len(axes):] or 1)) or [axes]
        labels = [label] * len(names) if isinstance(label, str) else list(label)
        for k, text in zip(names, labels):
            if text is not None:
                fig.layout[k].title.text = text
    return fig


def _title(title, kind, name):
    return f"PPC: {kind} — predictions of {name}" if title is None else title


def _interval_bounds(y, yrep, x, prob, prob_outer):
    """Rows sorted by x and the type-7 quantiles bayesplot's
    ``ppc_intervals_data`` draws: (x, y, ll, l, m, h, hh)."""
    y, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)
    if prob > prob_outer:          # bayesplot swaps them too
        prob, prob_outer = prob_outer, prob
    x = np.arange(len(y)) if x is None else np.asarray(x).flatten()
    order = np.argsort(x, kind="stable")
    x, y, yrep = x[order], y[order], yrep[:, order]

    def q(p):
        return np.percentile(yrep, 100 * p, axis=0)
    return (x, y, q((1 - prob_outer) / 2), q((1 - prob) / 2), q(0.5),
            q((1 + prob) / 2), q((1 + prob_outer) / 2))


def _segments(x, lo, hi):
    """Vertical segments x_i: lo_i -> hi_i as one line trace (None-separated)."""
    n = len(x)
    xs = np.empty(3 * n, dtype=object)
    ys = np.empty(3 * n, dtype=object)
    xs[0::3], xs[1::3], xs[2::3] = x, x, None
    ys[0::3], ys[1::3], ys[2::3] = lo, hi, None
    return list(xs), list(ys)


# =============================================================================
# Distributions
# =============================================================================

def ppc_density(y, yrep, n=50, bw_adjust=1.0, title=None, name="y", x_label=None, y_label=None):
    """KDE of y (black) vs n random draws from yrep rows (blue).

    Args:
        y: Observed data, shape (n_obs,).
        yrep: Posterior predictive matrix, shape (n_rep, n_obs).
        n: Max number of yrep lines to show.
        bw_adjust: Bandwidth multiplier passed to scipy KDE.
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels ("Predictions of
            <name>"); m.diag passes the observed site name.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
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
            showlegend=(i == 0), name=_preds(name),
        ))

    y_clean = np.asarray(y, dtype=np.float64).flatten()
    y_clean = y_clean[np.isfinite(y_clean)]
    if len(y_clean) >= 2 and float(np.std(y_clean)) > 1e-12:
        kde_y = _kde(stats, y_clean, bw_adjust)
        fig.add_trace(go.Scatter(
            x=xs, y=kde_y(xs), mode="lines",
            line=dict(color=_OBS_COLOR, width=2.5),
            name=_obs(name),
        ))

    fig.update_layout(
        title=_title(title, "Density overlay", name), xaxis_title=name,
        yaxis_title="Density",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return _axis_labels(fig, x_label, y_label)


def ppc_hist(y, yrep, n=8, bins=30, title=None, name="y", x_label=None, y_label=None):
    """Histograms: observed y plus n random yrep draws.

    Args:
        y: Observed data.
        yrep: Posterior predictive matrix.
        n: Number of yrep draws to show.
        bins: Number of histogram bins.
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
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
    subtitles = [_obs(name)] + [f"Prediction {i + 1}" for i in range(len(idx))]
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

    fig.update_layout(title=_title(title, "Histograms", name),
                      height=280 * nrows, showlegend=False, plot_bgcolor="white")
    return _axis_labels(fig, x_label, y_label)


def ppc_boxplot(y, yrep, n=20, title=None, name="y", x_label=None, y_label=None):
    """Box-and-whisker: y (black) vs n yrep draws.

    Args:
        y: Observed data.
        yrep: Posterior predictive matrix.
        n: Number of yrep draws.
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
    Returns:
        plotly Figure.
    """
    y, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(yrep), size=min(n, len(yrep)), replace=False)

    fig = go.Figure()
    fig.add_trace(go.Box(y=y, marker_color=_OBS_COLOR,
                         name="observed", showlegend=False))
    for i, s in enumerate(idx):
        fig.add_trace(go.Box(y=yrep[s], marker_color=_REP_COLOR_SOLID,
                             showlegend=False, name=f"pred {i + 1}"))
    fig.update_layout(title=_title(title, "Boxplots", name), yaxis_title=name,
                      xaxis_title=f"Observed {name} and {_preds(name).lower()}",
                      plot_bgcolor="white", boxmode="overlay")
    return _axis_labels(fig, x_label, y_label)


# =============================================================================
# Test statistics
# =============================================================================

def ppc_stat(y, yrep, stat="mean", title=None, name="y", x_label=None, y_label=None):
    """Distribution of test statistic over yrep vs observed value.

    Args:
        y: Observed data.
        yrep: Posterior predictive matrix.
        stat: Statistic name or callable. Options: 'mean', 'median', 'sd',
              'var', 'min', 'max', 'q25', 'q75', 'skew', 'n_zero', 'n_pos'.
              A callable takes one 1D array and returns a number, e.g.
              ``lambda v: np.percentile(v, 90)``.
        title: Plot title. Defaults to one naming the statistic, the outcome
            and the Bayesian p-values.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
    Returns:
        plotly Figure with Bayesian p-value annotation.
    """
    import scipy.stats as scipy_stats
    y, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)
    fn = _stat_fn(stat)
    stat_name = stat if isinstance(stat, str) else getattr(stat, "__name__", "T")
    if stat_name == "<lambda>":
        stat_name = "T"

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
        name=f"{stat_name} of each prediction of {name}",
    ))
    fig.add_vline(x=t_y, line_color=_OBS_COLOR, line_width=2.5,
                  annotation_text=f"observed {stat_name} = {t_y:.3g}",
                  annotation_position="top right")
    fig.update_layout(
        title=title or (f"PPC: {stat_name} of {name}  |  "
                        f"P(predicted ≥ observed) = {pval:.3f}"
                        f"  (two-sided p = {pval_2s:.3f})"),
        xaxis_title=f"{stat_name}({name})",
        yaxis_title="Density",
        plot_bgcolor="white",
    )
    return _axis_labels(fig, x_label, y_label)


def ppc_stat_2d(y, yrep, stat1="mean", stat2="sd", title=None, name="y", x_label=None, y_label=None):
    """Scatter of two test statistics over yrep vs observed point.

    Args:
        y: Observed data.
        yrep: Posterior predictive matrix.
        stat1, stat2: Statistic names or callables.
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
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
        name=_preds(name),
    ))
    fig.add_trace(go.Scatter(
        x=[t1_y], y=[t2_y], mode="markers",
        marker=dict(color=_OBS_COLOR, size=12, symbol="x"),
        name=_obs(name),
    ))
    fig.update_layout(
        title=title or f"PPC: {s1_name} vs {s2_name} — predictions of {name}",
        xaxis_title=f"{s1_name}({name})", yaxis_title=f"{s2_name}({name})",
        plot_bgcolor="white",
    )
    return _axis_labels(fig, x_label, y_label)


# =============================================================================
# Intervals
# =============================================================================

def ppc_intervals(y, yrep, x=None, prob=0.5, prob_outer=0.9, title=None,
                  name="y", x_label=None, y_label=None):
    """Predictive interval of each observation, with the observed value.

    One vertical bar per observation, as bayesplot's ``ppc_intervals``: a
    thin line for the outer (prob_outer) interval, a thick line for the inner
    (prob) interval, a light point for the median prediction and a dark
    point for the observed value. About prob of the observed points should
    fall on the thick bars and prob_outer on the thin ones.

    Args:
        y: Observed data, shape (n_obs,).
        yrep: Posterior predictive matrix, shape (n_rep, n_obs).
        x: Optional x-axis variable, shape (n_obs,) (e.g. a predictor).
            Defaults to the observation index.
        prob: Inner interval probability (e.g., 0.5).
        prob_outer: Outer interval probability (e.g., 0.9).
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
    Returns:
        plotly Figure.
    """
    x_was_none = x is None
    x, y, ll, l, m, h, hh = _interval_bounds(y, yrep, x, prob, prob_outer)
    prob, prob_outer = sorted((prob, prob_outer))

    fig = go.Figure()
    xs, ys = _segments(x, ll, hh)
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", line=dict(color=_REP_COLOR_SOLID, width=1.5),
        name=f"{int(round(prob_outer * 100))}% interval", hoverinfo="skip",
    ))
    xs, ys = _segments(x, l, h)
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", line=dict(color="cornflowerblue", width=5),
        name=f"{int(round(prob * 100))}% interval", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=m, mode="markers",
        marker=dict(color="white", size=7,
                    line=dict(color="cornflowerblue", width=1.5)),
        name=f"Median prediction of {name}",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(color=_OBS_COLOR, size=5),
        name=_obs(name),
    ))
    fig.update_layout(title=_title(title, "Predictive intervals", name),
                      xaxis_title="Observation" if x_was_none else "x",
                      yaxis_title=name, plot_bgcolor="white")
    return _axis_labels(fig, x_label, y_label)


def ppc_ribbon(y, yrep, x=None, prob=0.5, prob_outer=0.9, title=None,
               name="y", x_label=None, y_label=None):
    """Predictive intervals drawn as continuous ribbons along x.

    Same intervals as :func:`ppc_intervals`, as bayesplot's ``ppc_ribbon``:
    a light outer ribbon, a darker inner ribbon, a thin line for the median
    prediction and the observed values as points. Easier to read than the
    bars when x is continuous (time, dose, a predictor).

    Args:
        y: Observed data, shape (n_obs,).
        yrep: Posterior predictive matrix, shape (n_rep, n_obs).
        x: Optional x-axis variable, shape (n_obs,). Defaults to the
            observation index.
        prob: Inner interval probability (e.g., 0.5).
        prob_outer: Outer interval probability (e.g., 0.9).
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
    Returns:
        plotly Figure.
    """
    x_was_none = x is None
    x, y, ll, l, m, h, hh = _interval_bounds(y, yrep, x, prob, prob_outer)
    prob, prob_outer = sorted((prob, prob_outer))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]), y=np.concatenate([hh, ll[::-1]]),
        fill="toself", fillcolor="rgba(100,149,237,0.2)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
        name=f"{int(round(prob_outer * 100))}% interval",
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]), y=np.concatenate([h, l[::-1]]),
        fill="toself", fillcolor="rgba(100,149,237,0.45)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
        name=f"{int(round(prob * 100))}% interval",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=m, mode="lines", line=dict(color="cornflowerblue", width=1),
        name=f"Median prediction of {name}",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", marker=dict(color=_OBS_COLOR, size=5),
        name=_obs(name),
    ))
    fig.update_layout(title=_title(title, "Predictive ribbon", name),
                      xaxis_title="Observation" if x_was_none else "x",
                      yaxis_title=name, plot_bgcolor="white")
    return _axis_labels(fig, x_label, y_label)


# =============================================================================
# Errors
# =============================================================================

def ppc_error_scatter(y, yrep, title=None, name="y", x_label=None, y_label=None):
    """Scatter of observed y vs posterior predictive mean E[yrep].

    Perfect model: points fall on y = x diagonal.

    Args:
        y: Observed data.
        yrep: Posterior predictive matrix.
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
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
    fig.update_layout(title=_title(title, "Observed vs mean prediction", name),
                      xaxis_title=f"Mean prediction of {name}",
                      yaxis_title=_obs(name), plot_bgcolor="white")
    return _axis_labels(fig, x_label, y_label)


def ppc_error_hist(y, yrep, bins=40, title=None, name="y", x_label=None, y_label=None):
    """Distribution of mean residuals r_i = y_i - E[yrep_i].

    Plots the mean residual per observation; bayesplot's ppc_error_hist
    instead shows the per-draw errors y - yrep[s, i] faceted by draw.

    Args:
        y: Observed data.
        yrep: Posterior predictive matrix.
        bins: Number of histogram bins.
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
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
    fig.update_layout(title=_title(title, "Residuals", name),
                      xaxis_title=f"Observed {name} − mean prediction",
                      yaxis_title="Count", plot_bgcolor="white")
    return _axis_labels(fig, x_label, y_label)


# =============================================================================
# Scatterplots
# =============================================================================

def ppc_scatter(y, yrep, n_reps=9, title=None, name="y", x_label=None, y_label=None):
    """Small-multiple scatterplots of y vs n_reps random rows of yrep.

    Args:
        y: Observed data.
        yrep: Posterior predictive matrix.
        n_reps: Number of replications to show.
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
    Returns:
        plotly Figure.
    """
    y, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(yrep), size=min(n_reps, len(yrep)), replace=False)

    ncols = 3
    nrows = int(np.ceil(len(idx) / ncols))
    subtitles = [f"Prediction {i + 1}" for i in range(len(idx))]
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
        fig.update_xaxes(title_text=f"predicted {name}", row=r, col=c)
        fig.update_yaxes(title_text=f"observed {name}", row=r, col=c)

    fig.update_layout(title=_title(title, "Observed vs individual predictions", name),
                      height=300 * nrows, plot_bgcolor="white")
    return _axis_labels(fig, x_label, y_label)


# =============================================================================
# Discrete
# =============================================================================

def ppc_rootogram(y, yrep, title=None, name="y", x_label=None, y_label=None):
    """Hanging rootogram for count outcomes.

    Compares sqrt of observed counts (bars) to expected sqrt counts (curve).
    Bars hanging below zero indicate over-prediction; above zero: under-prediction.

    Args:
        y: Observed integer counts.
        yrep: Posterior predictive matrix of counts.
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
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
        name=f"√Expected count ({_preds(name).lower()})",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_layout(
        title=_title(title, "Rootogram", name), xaxis_title=name,
        yaxis_title="√Frequency",
        plot_bgcolor="white",
    )
    return _axis_labels(fig, x_label, y_label)


def ppc_bars(y, yrep, title=None, name="y", x_label=None, y_label=None):
    """Bar chart comparing observed y to posterior predictive yrep.

    For ordinal, categorical, or discrete outcomes.
    Shows observed counts and, per category, the median predicted count with
    a 90% interval, as in bayesplot's ppc_bars.

    Args:
        y: Observed discrete values.
        yrep: Posterior predictive matrix.
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
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

    rep_med  = np.median(rep_counts, axis=0)
    rep_lo   = np.percentile(rep_counts, 5, axis=0)
    rep_hi   = np.percentile(rep_counts, 95, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cats, y=obs_counts,
        marker_color=_OBS_COLOR, opacity=0.8,
        name=_obs(name),
    ))
    fig.add_trace(go.Scatter(
        x=cats, y=rep_med, mode="markers",
        marker=dict(color="cornflowerblue", size=10, symbol="diamond"),
        error_y=dict(type="data", symmetric=False,
                     array=rep_hi - rep_med,
                     arrayminus=rep_med - rep_lo),
        name=f"{_preds(name)}: median count ± 90% interval",
    ))
    fig.update_layout(title=_title(title, "Bar chart", name), xaxis_title=name,
                      yaxis_title="Count", plot_bgcolor="white")
    return _axis_labels(fig, x_label, y_label)


# =============================================================================
# LOO-based PPC
# =============================================================================

def _pit_density(pit, xs):
    """KDE of values in [0, 1] (Scott bandwidth) with reflection at 0 and 1.

    A plain KDE leaks mass outside [0, 1], so even perfectly uniform PIT
    values would draw a curve that drops to ~0.5 at both ends -- the shape
    of an over-dispersed model. Reflecting the sample (bayesplot's
    boundary correction) removes that artefact.
    """
    pit = np.asarray(pit, dtype=np.float64)
    n = len(pit)
    bw = np.std(pit, ddof=1) * n ** (-0.2)
    pts = np.concatenate([pit, -pit, 2.0 - pit])
    z = (np.asarray(xs, dtype=np.float64)[:, None] - pts[None, :]) / bw
    return np.exp(-0.5 * z * z).sum(axis=1) / (n * bw * np.sqrt(2.0 * np.pi))


def ppc_loo_pit(y, yrep, log_likelihood=None, reff=None, n_unif=100,
                title=None, name="y", x_label=None, y_label=None):
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
        n_unif: Number of reference curves, each the density of n_obs
            Uniform(0, 1) draws, drawn in grey: how far from flat a curve
            from a perfectly calibrated model can be by chance.
        title: Plot title. Defaults to one naming the outcome and saying
            whether the PIT is leave-one-out.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
    Returns:
        plotly Figure. Both densities are boundary-corrected (reflected at 0
        and 1), as in bayesplot.
    """
    import warnings
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

    kind = "LOO-PIT" if log_likelihood is not None else "PIT (not leave-one-out)"
    title = _title(title, kind, name)
    pit = pit[np.isfinite(pit)]
    if len(pit) < 2:
        return go.Figure().update_layout(title=title + " (insufficient data)")

    xs = np.linspace(0, 1, 300)
    fig = go.Figure()
    rng = np.random.default_rng(0)
    for j in range(int(n_unif)):
        fig.add_trace(go.Scatter(
            x=xs, y=_pit_density(rng.uniform(size=len(pit)), xs), mode="lines",
            line=dict(color="rgba(150,150,150,0.25)", width=1),
            hoverinfo="skip", legendgroup="unif", showlegend=(j == 0),
            name=f"Uniform samples of size {len(pit)}",
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[1, 1], mode="lines",
        line=dict(color="black", dash="dash", width=1.5),
        name="Uniform(0,1)",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=_pit_density(pit, xs), mode="lines",
        line=dict(color="cornflowerblue", width=2.5),
        name=f"{kind} of {name}",
    ))
    fig.update_layout(
        title=title, xaxis_title=kind, yaxis_title="Density",
        xaxis=dict(range=[0, 1]),
        plot_bgcolor="white",
    )
    return _axis_labels(fig, x_label, y_label)


def ppc_intervals_sorted(y, yrep, x=None, prob=0.5, prob_outer=0.9,
                         title=None, name="y", x_label=None, y_label=None):
    """Predictive intervals with the observations ordered by observed value.

    This is :func:`ppc_intervals` with observation i placed at the rank of
    y_i, so the observed points rise from left to right and systematic
    over/under-prediction across the range of the outcome is easy to read.
    It performs no leave-one-out computation (``ppc_loo_intervals`` is a
    deprecated alias).

    Args:
        y: Observed data.
        yrep: Posterior predictive matrix.
        x: Optional ordering variable. Defaults to the rank of y.
        prob: Inner interval probability (e.g., 0.5).
        prob_outer: Outer interval probability (e.g., 0.9).
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title. Defaults to the plot's own label.
        y_label: Y-axis title. Defaults to the outcome name.
    Returns:
        plotly Figure.
    """
    sort_by_y = x is None
    if sort_by_y:
        # The rank of each observation; argsort(y) alone is the inverse
        # permutation and left the observations unsorted.
        x = np.argsort(np.argsort(np.asarray(y, dtype=float).flatten(),
                                  kind="stable"), kind="stable")
    fig = ppc_intervals(y, yrep, x=x, prob=prob, prob_outer=prob_outer,
                        title=_title(title, "Predictive intervals sorted by observed value", name),
                        name=name)
    if sort_by_y:
        fig.update_layout(xaxis_title=f"Observations ranked by observed {name}")
    return _axis_labels(fig, x_label, y_label)


def ppc_loo_intervals(y, yrep, x=None, prob=0.9, title=None, name="y",
                      x_label=None, y_label=None):
    """Deprecated alias for :func:`ppc_intervals_sorted` (it is not LOO-based).

    ``prob`` is the outer interval, as before; the inner one is 50%.
    """
    import warnings
    warnings.warn(
        "ppc_loo_intervals performs no leave-one-out computation; it is "
        "ppc_intervals sorted by y. Use ppc_intervals_sorted instead.",
        DeprecationWarning, stacklevel=2,
    )
    return ppc_intervals_sorted(y, yrep, x=x, prob=min(0.5, prob),
                                prob_outer=prob, title=title, name=name,
                                x_label=x_label, y_label=y_label)
