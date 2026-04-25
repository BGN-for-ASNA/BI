"""
JAX+Plotly diagnostic methods for BI models.

Two entry points:
  patch_diag_class(cls)       — patches a tool class; methods take (self, m, ...)
  bind_diag_to_model(diag, m) — patches a diag *instance*; methods take (include=, exclude=, ...)
                                 and close over m, so m.diag.density() respects m.posteriors
"""
import BI.Diagnostic.jax_diagnostics as jd
from BI.Diagnostic.jax_diagnostics import iter_expanded, filter_posterior_dict
import plotly.colors as pcolors

_COLORS = pcolors.qualitative.Plotly


# =============================================================================
# Shared helpers
# =============================================================================

def _get_posteriors(m, by_chain=False):
    if by_chain and hasattr(m, 'posteriors_by_chain'):
        return m.posteriors_by_chain
    if hasattr(m, 'posteriors'):
        return m.posteriors
    if hasattr(m, 'posterior_samples'):
        return m.posterior_samples
    if isinstance(m, dict):
        return m
    raise AttributeError(
        "Cannot find posteriors. Expected m.posteriors, m.posterior_samples, or a dict.")


def _source(m, filtered=True, by_chain=False):
    """Select posteriors dict: filtered=False uses posteriors_full (ignores active filter)."""
    if not filtered:
        if by_chain:
            full = getattr(m, 'posteriors_by_chain_full', None)
        else:
            full = getattr(m, 'posteriors_full', None)
        if full is not None:
            return full
    return _get_posteriors(m, by_chain=by_chain)


def _expand(m, include=None, exclude=None, filter_regex=None, filtered=True):
    """Return list of (label, 1D_samples).

    filtered=True  — uses m.posteriors (active filter) plus per-call include/exclude.
    filtered=False — uses m.posteriors_full (all parameters) plus per-call include/exclude.
    filter_regex applied on expanded labels.
    """
    posteriors = filter_posterior_dict(_source(m, filtered=filtered),
                                       include=include, exclude=exclude)
    return list(iter_expanded(posteriors, filter_regex=filter_regex))


# =============================================================================
# Module-level plot functions  (take m directly)
# =============================================================================

def _plot_trace(m, include=None, exclude=None, filter_regex=None, filtered=True):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    expanded = _expand(m, include, exclude, filter_regex, filtered)
    titles = [f'{name} {s}' for (name, _) in expanded for s in ['Trace', 'Posterior']]
    fig = make_subplots(rows=len(expanded), cols=2, subplot_titles=titles)
    for i, (name, samples) in enumerate(expanded):
        color = _COLORS[i % len(_COLORS)]
        fig.add_trace(go.Scatter(y=samples, mode='lines', name=name,
                                 line=dict(color=color), showlegend=False), row=i+1, col=1)
        fig.add_trace(go.Histogram(x=samples, name=name, marker_color=color,
                                   showlegend=False, opacity=0.7, nbinsx=50), row=i+1, col=2)
    fig.update_layout(height=300*len(expanded), title_text="Trace and Posterior Plots",
                      barmode='overlay')
    return fig


def _plot_posterior(m, include=None, exclude=None, filter_regex=None,
                    figsize=(800, 400), hdi_prob=0.94, filtered=True):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    expanded = _expand(m, include, exclude, filter_regex, filtered)
    fig = make_subplots(rows=1, cols=len(expanded), subplot_titles=[e[0] for e in expanded])
    for i, (name, samples) in enumerate(expanded):
        color = _COLORS[i % len(_COLORS)]
        fig.add_trace(go.Histogram(x=samples, name=name, marker_color=color,
                                   showlegend=False, opacity=0.7, nbinsx=50), row=1, col=i+1)
        mean_val = float(np.mean(samples))
        hdi_vals = jd.hdi(samples, hdi_prob=hdi_prob)
        fig.add_vline(x=mean_val, line_dash="dash", line_color="black", row=1, col=i+1)
        fig.add_vline(x=float(hdi_vals[0]), line_dash="dot", line_color="firebrick", row=1, col=i+1)
        fig.add_vline(x=float(hdi_vals[1]), line_dash="dot", line_color="firebrick", row=1, col=i+1)
    n = len(expanded)
    fig.update_layout(title_text=f"Posterior Distributions ({hdi_prob*100:.0f}% HDI)",
                      width=figsize[0] if n < 4 else figsize[0]*n//3,
                      height=figsize[1], barmode='overlay')
    return fig


def _plot_autocor(m, include=None, exclude=None, filter_regex=None, max_lag=40, filtered=True):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    expanded = _expand(m, include, exclude, filter_regex, filtered)
    fig = make_subplots(rows=len(expanded), cols=1,
                        subplot_titles=[f"Autocorrelation of {e[0]}" for e in expanded])
    for i, (name, samples) in enumerate(expanded):
        samples = np.asarray(samples)
        autocorr = [1.0] + [float(np.corrcoef(samples[:-t], samples[t:])[0, 1])
                            for t in range(1, max_lag)]
        color = _COLORS[i % len(_COLORS)]
        fig.add_trace(go.Bar(y=autocorr, name=name, marker_color=color, showlegend=False),
                      row=i+1, col=1)
    fig.update_layout(height=250*len(expanded), title_text="Autocorrelation Plots",
                      barmode='group')
    return fig


def _plot_forest(m, include=None, exclude=None, filter_regex=None, hdi_prob=0.95, filtered=True):
    import plotly.graph_objects as go
    import jax.numpy as jnp

    expanded = _expand(m, include, exclude, filter_regex, filtered)
    fig = go.Figure()
    for i, (name, samples) in enumerate(expanded):
        color = _COLORS[i % len(_COLORS)]
        fig.add_trace(go.Violin(x=samples, y=[f" {name} "], name=name, legendgroup=name,
                                orientation='h', side='both', points=False, fillcolor=color,
                                opacity=0.4, line_width=0, spanmode='hard'))
        mean_val = float(jnp.mean(samples))
        hdi_vals = jd.hdi(samples, hdi_prob=hdi_prob)
        lo, hi = float(hdi_vals[0]), float(hdi_vals[1])
        fig.add_trace(go.Scatter(x=[mean_val], y=[f" {name} "], mode='markers',
                                 legendgroup=name, name=name,
                                 marker=dict(color=color, size=8),
                                 error_x=dict(type='data', symmetric=False,
                                              array=[hi-mean_val], arrayminus=[mean_val-lo],
                                              width=4, color=color),
                                 showlegend=False))
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.update_layout(title_text=f'Forest Plot ({hdi_prob*100:.1f}% HDI)',
                      xaxis_title="Parameter Value", yaxis_title="Parameter",
                      violingap=0.1, plot_bgcolor='white')
    fig.update_yaxes(autorange="reversed")
    return fig


def _plot_density(m, include=None, exclude=None, filter_regex=None, shade=0.4, filtered=True):
    import plotly.graph_objects as go
    import plotly.colors as pcolors_
    from plotly.subplots import make_subplots
    import seaborn as sns
    import matplotlib.pyplot as plt

    expanded = _expand(m, include, exclude, filter_regex, filtered)
    fig = make_subplots(rows=len(expanded), cols=1,
                        subplot_titles=[f"Density of {e[0]}" for e in expanded])
    for i, (name, samples) in enumerate(expanded):
        color = _COLORS[i % len(_COLORS)]
        rgb = pcolors_.hex_to_rgb(color)
        fill = f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{shade})'
        with sns.plotting_context(rc={"figure.figsize": (1, 1)}):
            kde_plot = sns.kdeplot(samples)
            kde = kde_plot.get_lines()[0].get_data()
            plt.close()
        fig.add_trace(go.Scatter(x=kde[0], y=kde[1], fill='tozeroy', mode='lines',
                                 name=name, showlegend=False, fillcolor=fill, line_color=color),
                      row=i+1, col=1)
    fig.update_layout(height=300*len(expanded), title_text="Density Plots")
    return fig


def _plot_pair(m, include=None, exclude=None, filter_regex=None,
               colorscale="Viridis", max_points=1000,
               point_color='rgba(40, 150, 200, 0.4)', filtered=True):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np

    expanded = _expand(m, include, exclude, filter_regex, filtered)
    labels = [e[0] for e in expanded]
    samples_dict = {name: np.asarray(s) for (name, s) in expanded}
    n_vars = len(labels)
    df = pd.DataFrame(samples_dict)
    plot_df = df.sample(n=max_points, random_state=42) if len(df) > max_points else df

    fig = make_subplots(rows=n_vars, cols=n_vars, horizontal_spacing=0.03, vertical_spacing=0.03)
    for i in range(n_vars):
        for j in range(n_vars):
            v1, v2 = labels[i], labels[j]
            if i == j:
                fig.add_trace(go.Histogram(x=df[v1], marker_color='#440154'), row=i+1, col=j+1)
            elif i > j:
                fig.add_trace(go.Histogram2dContour(
                    x=df[v2], y=df[v1], colorscale=colorscale, showscale=False,
                    contours=dict(coloring='lines'), line=dict(width=1)), row=i+1, col=j+1)
                fig.add_trace(go.Scatter(x=plot_df[v2], y=plot_df[v1], mode='markers',
                                         marker=dict(size=3, color=point_color)), row=i+1, col=j+1)
                fig.add_trace(go.Scatter(x=[df[v2].median()], y=[df[v1].median()], mode='markers',
                                         marker=dict(symbol='square', color='black', size=8)),
                              row=i+1, col=j+1)
    fig.update_layout(title_text="Pair Plot: Histograms, Density, and Samples",
                      height=250*n_vars, width=250*n_vars, showlegend=False, plot_bgcolor='white')
    for i in range(n_vars):
        fig.update_yaxes(title_text=labels[i], row=i+1, col=1,
                         showline=True, linewidth=1, linecolor='black', mirror=True)
    for j in range(n_vars):
        fig.update_xaxes(title_text=labels[j], row=n_vars, col=j+1,
                         showline=True, linewidth=1, linecolor='black', mirror=True)
    return fig


# =============================================================================
# patch_diag_class  — patches a tool class (methods take self + m)
# =============================================================================

def patch_diag_class(cls):
    """Patch cls so all diagnostic/plot methods take (self, m, include=, exclude=, ...)."""

    def para_names(self, m):
        return list(_get_posteriors(m).keys())

    def summary_jax(self, m, round_to=2, hdi_prob=0.89,
                    include=None, exclude=None, filter_regex=None,
                    var_names=None, exclude_vars=None):
        has_chains = hasattr(m, 'posteriors_by_chain')
        self.tab_summary = jd.summary(
            _get_posteriors(m, by_chain=has_chains),
            round_to=round_to, hdi_prob=hdi_prob,
            include=include or var_names, exclude=exclude or exclude_vars,
            filter_regex=filter_regex, group_by_chain=has_chains)
        return self.tab_summary

    def rhat_jax(self, m, include=None, exclude=None,
                 var_names=None, exclude_vars=None):
        return jd.rhat(_get_posteriors(m, by_chain=True),
                       include=include or var_names, exclude=exclude or exclude_vars)

    def ess_jax(self, m, include=None, exclude=None, kind="bulk",
                var_names=None, exclude_vars=None):
        has_chains = hasattr(m, 'posteriors_by_chain')
        return jd.ess(_get_posteriors(m, by_chain=has_chains),
                      include=include or var_names, exclude=exclude or exclude_vars, kind=kind)

    def mcse_jax(self, m, include=None, exclude=None,
                 var_names=None, exclude_vars=None):
        has_chains = hasattr(m, 'posteriors_by_chain')
        return jd.mcse(_get_posteriors(m, by_chain=has_chains),
                       include=include or var_names, exclude=exclude or exclude_vars)

    def plot_trace_jax(self, m, include=None, exclude=None, filter_regex=None):
        return _plot_trace(m, include, exclude, filter_regex)

    def posterior_jax(self, m, include=None, exclude=None, filter_regex=None,
                      figsize=(800, 400), hdi_prob=0.94):
        return _plot_posterior(m, include, exclude, filter_regex, figsize, hdi_prob)

    def autocor_jax(self, m, include=None, exclude=None, filter_regex=None, max_lag=40):
        return _plot_autocor(m, include, exclude, filter_regex, max_lag)

    def forest_jax(self, m, include=None, exclude=None, filter_regex=None, hdi_prob=0.95):
        return _plot_forest(m, include, exclude, filter_regex, hdi_prob)

    def density_jax(self, m, include=None, exclude=None, filter_regex=None, shade=0.4):
        return _plot_density(m, include, exclude, filter_regex, shade)

    def pair_jax(self, m, include=None, exclude=None, filter_regex=None,
                 colorscale="Viridis", max_points=1000,
                 point_color='rgba(40, 150, 200, 0.4)'):
        return _plot_pair(m, include, exclude, filter_regex, colorscale, max_points, point_color)

    def model_checks_jax(self, m, include=None, exclude=None, filter_regex=None):
        kw = dict(include=include, exclude=exclude, filter_regex=filter_regex)
        print("Posterior Plots:");      self.posterior(m, **kw).show()
        print("\nAutocorrelation:");    self.autocor(m, **kw).show()
        print("\nTrace Plots:");        self.plot_trace(m, **kw).show()
        print("\nForest Plot:");        self.forest(m, **kw).show()
        print("\nPair Plot:");          self.pair(m, **kw).show()

    def compute_ll(self, m, model, *args, obs_name=None, **kwargs):
        return jd.compute_log_likelihood(model, _get_posteriors(m), *args,
                                          obs_name=obs_name, **kwargs)

    def loo_jax(self, m=None, model=None, *args, log_likelihood=None,
                obs_name=None, pointwise=False, scale="log", **kwargs):
        if log_likelihood is None:
            if m is None or model is None:
                raise ValueError("Pass m+model+data args or log_likelihood=.")
            log_likelihood = jd.compute_log_likelihood(
                model, _get_posteriors(m), *args, obs_name=obs_name, **kwargs)
        return jd.loo(log_likelihood, pointwise=pointwise, scale=scale)

    def waic_jax(self, m=None, model=None, *args, log_likelihood=None,
                 obs_name=None, pointwise=False, scale="log", **kwargs):
        if log_likelihood is None:
            if m is None or model is None:
                raise ValueError("Pass m+model+data args or log_likelihood=.")
            log_likelihood = jd.compute_log_likelihood(
                model, _get_posteriors(m), *args, obs_name=obs_name, **kwargs)
        return jd.waic(log_likelihood, pointwise=pointwise, scale=scale)

    cls.para_names           = para_names
    cls.summary              = summary_jax
    cls.rhat                 = rhat_jax
    cls.ess                  = ess_jax
    cls.mcse                 = mcse_jax
    cls.plot_trace           = plot_trace_jax
    cls.posterior            = posterior_jax
    cls.autocor              = autocor_jax
    cls.forest               = forest_jax
    cls.density              = density_jax
    cls.pair                 = pair_jax
    cls.model_checks         = model_checks_jax
    cls.loo                  = loo_jax
    cls.WAIC                 = waic_jax
    cls.compute_log_likelihood = compute_ll
    cls.compare              = staticmethod(jd.compare)
    return cls


# =============================================================================
# bind_diag_to_model — patches a diag *instance* to use m.posteriors directly
# =============================================================================

def bind_diag_to_model(diag_obj, m):
    """Replace diag_obj's plot/diagnostic methods with JAX+plotly versions.

    All methods accept:
      filtered=True  (default) — use m.posteriors (respects active filter)
      filtered=False           — use m.posteriors_full (all parameters)
    Per-call include/exclude further restrict on top of whichever source is chosen.
    """
    def _summary(include=None, exclude=None, round_to=2, hdi_prob=0.89,
                 filter_regex=None, filtered=True):
        src = _source(m, filtered=filtered, by_chain=True)
        return jd.summary(src, include=include, exclude=exclude,
                          filter_regex=filter_regex, round_to=round_to,
                          hdi_prob=hdi_prob, group_by_chain=True)

    def _rhat(include=None, exclude=None, filtered=True):
        return jd.rhat(_source(m, filtered=filtered, by_chain=True),
                       include=include, exclude=exclude)

    def _ess(include=None, exclude=None, kind="bulk", filtered=True):
        return jd.ess(_source(m, filtered=filtered, by_chain=True),
                      include=include, exclude=exclude, kind=kind)

    def _mcse(include=None, exclude=None, filtered=True):
        return jd.mcse(_source(m, filtered=filtered, by_chain=True),
                       include=include, exclude=exclude)

    def _model_checks(include=None, exclude=None, filter_regex=None, filtered=True):
        kw = dict(include=include, exclude=exclude, filter_regex=filter_regex, filtered=filtered)
        print("Posterior Plots:");   diag_obj.posterior(**kw).show()
        print("\nAutocorrelation:"); diag_obj.autocor(**kw).show()
        print("\nTrace Plots:");     diag_obj.plot_trace(**kw).show()
        print("\nForest Plot:");     diag_obj.forest(**kw).show()
        print("\nPair Plot:");       diag_obj.pair(**kw).show()

    diag_obj.summary      = _summary
    diag_obj.rhat         = _rhat
    diag_obj.ess          = _ess
    diag_obj.mcse         = _mcse
    diag_obj.plot_trace   = lambda include=None, exclude=None, filtered=True, **kw: _plot_trace(m, include, exclude, filtered=filtered, **kw)
    diag_obj.posterior    = lambda include=None, exclude=None, filtered=True, **kw: _plot_posterior(m, include, exclude, filtered=filtered, **kw)
    diag_obj.autocor      = lambda include=None, exclude=None, filtered=True, **kw: _plot_autocor(m, include, exclude, filtered=filtered, **kw)
    diag_obj.forest       = lambda include=None, exclude=None, filtered=True, **kw: _plot_forest(m, include, exclude, filtered=filtered, **kw)
    diag_obj.density      = lambda include=None, exclude=None, filtered=True, **kw: _plot_density(m, include, exclude, filtered=filtered, **kw)
    diag_obj.pair         = lambda include=None, exclude=None, filtered=True, **kw: _plot_pair(m, include, exclude, filtered=filtered, **kw)
    diag_obj.model_checks = _model_checks
