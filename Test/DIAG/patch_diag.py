"""
Drop-in integration: patch the diag class so every diagnostic/plot takes a
fitted model `m` with `m.posteriors` as the sample dict.

Layout
------
`m.posteriors` is a dict of {name: array} where each array has shape
(n_samples, ...) — first dim is always samples, remaining dims are the
parameter shape. No separate chain dimension.

API
---
Every method takes `m` as first argument:

    d = diagWIP(...)          # stateless diag tool
    d.summary(m)              # returns pandas DataFrame
    d.para_names(m)           # raw dict keys: ["a", "b", "rho", ...]
    d.rhat(m)
    d.ess(m)
    d.plot_trace(m)
    d.posterior(m)
    d.autocor(m)
    d.forest(m)
    d.density(m)
    d.pair(m)
    d.model_checks(m)
    d.loo(m, model, X, y)     # model + data args for log-lik
    d.WAIC(m, model, X, y)

Filtering
---------
All methods accept var_names / exclude_vars / filter_regex. Filtering
operates on EXPANDED names so "rho" selects all rho[i,j,...] but never
matches "rho_var1".

    d.summary(m, var_names=["rho"])
    d.plot_trace(m, exclude_vars="sigma")
    d.forest(m, filter_regex=r"^beta")
"""
import jax_diagnostics as jd
from jax_diagnostics import iter_expanded


def _get_posteriors(m, by_chain=False):
    """Extract the posterior-samples dict from `m`.

    by_chain=True: returns chain-structured (n_chains, n_draws, ...) when available.
    by_chain=False: returns flat (n_draws, ...) samples.
    """
    if by_chain and hasattr(m, 'posteriors_by_chain'):
        return m.posteriors_by_chain
    if hasattr(m, 'posteriors'):
        return m.posteriors
    if hasattr(m, 'posterior_samples'):
        return m.posterior_samples
    if isinstance(m, dict):
        return m
    raise AttributeError(
        "Cannot find posteriors on the passed object. Expected `m.posteriors` "
        "(a dict of JAX arrays), `m.posterior_samples`, or a dict.")


def patch_diag_class(cls):
    """Patch the diag class so methods take `m` and operate on `m.posteriors`.

    Replaces all diagnostic and plot methods on the class.
    """
    import plotly.colors as pcolors
    _COLORS = pcolors.qualitative.Plotly

    # ------------------------------------------------------------------
    # para_names — raw dict keys, no shape expansion
    # ------------------------------------------------------------------
    def para_names(self, m):
        """List of parameter names (raw dict keys, no index expansion).

        For {"a": (N,), "rho": (N, 2, 3), "sigma": (N,)} returns
        ["a", "rho", "sigma"].
        """
        return list(_get_posteriors(m).keys())

    cls.para_names = para_names

    # ------------------------------------------------------------------
    # summary / rhat / ess / mcse
    # ------------------------------------------------------------------
    def summary_jax(self, m, round_to=2, hdi_prob=0.89,
                    include=None, exclude=None, filter_regex=None,
                    var_names=None, exclude_vars=None):
        has_chains = hasattr(m, 'posteriors_by_chain')
        self.tab_summary = jd.summary(
            _get_posteriors(m, by_chain=has_chains),
            round_to=round_to, hdi_prob=hdi_prob,
            include=include or var_names,
            exclude=exclude or exclude_vars,
            filter_regex=filter_regex,
            group_by_chain=has_chains)
        return self.tab_summary

    def rhat_jax(self, m, include=None, exclude=None,
                 var_names=None, exclude_vars=None):
        return jd.rhat(_get_posteriors(m, by_chain=True),
                       include=include or var_names,
                       exclude=exclude or exclude_vars)

    def ess_jax(self, m, include=None, exclude=None, kind="bulk",
                var_names=None, exclude_vars=None):
        has_chains = hasattr(m, 'posteriors_by_chain')
        return jd.ess(_get_posteriors(m, by_chain=has_chains),
                      include=include or var_names,
                      exclude=exclude or exclude_vars,
                      kind=kind)

    def mcse_jax(self, m, include=None, exclude=None,
                 var_names=None, exclude_vars=None):
        has_chains = hasattr(m, 'posteriors_by_chain')
        return jd.mcse(_get_posteriors(m, by_chain=has_chains),
                       include=include or var_names,
                       exclude=exclude or exclude_vars)

    # ------------------------------------------------------------------
    # Internal helper: expand+filter, yield (label, 1D samples)
    # ------------------------------------------------------------------
    def _expand(self, m, var_names, exclude_vars, filter_regex):
        return list(iter_expanded(
            _get_posteriors(m), var_names, exclude_vars, filter_regex))

    cls._expand = _expand

    # ------------------------------------------------------------------
    # plot_trace (single chain, so just line + histogram per param)
    # ------------------------------------------------------------------
    def plot_trace_jax(self, m, var_names=None, exclude_vars=None,
                       filter_regex=None):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        expanded = self._expand(m, var_names, exclude_vars, filter_regex)
        titles = [f'{name} {s}' for (name, _) in expanded
                  for s in ['Trace', 'Posterior']]
        fig = make_subplots(rows=len(expanded), cols=2,
                            subplot_titles=titles)

        for i, (name, samples) in enumerate(expanded):
            color = _COLORS[i % len(_COLORS)]
            fig.add_trace(
                go.Scatter(y=samples, mode='lines', name=name,
                           line=dict(color=color), showlegend=False),
                row=i + 1, col=1)
            fig.add_trace(
                go.Histogram(x=samples, name=name, marker_color=color,
                             showlegend=False, opacity=0.7, nbinsx=50),
                row=i + 1, col=2)

        fig.update_layout(height=300 * len(expanded),
                          title_text="Trace and Posterior Plots",
                          barmode='overlay')
        return fig

    # ------------------------------------------------------------------
    # posterior — histograms with mean + HDI lines
    # ------------------------------------------------------------------
    def posterior_jax(self, m, var_names=None, exclude_vars=None,
                      filter_regex=None, figsize=(800, 400), hdi_prob=0.94):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np

        expanded = self._expand(m, var_names, exclude_vars, filter_regex)
        fig = make_subplots(rows=1, cols=len(expanded),
                            subplot_titles=[e[0] for e in expanded])

        for i, (name, samples) in enumerate(expanded):
            color = _COLORS[i % len(_COLORS)]
            fig.add_trace(
                go.Histogram(x=samples, name=name, marker_color=color,
                             showlegend=False, opacity=0.7, nbinsx=50),
                row=1, col=i + 1)
            mean_val = float(np.mean(samples))
            hdi_vals = jd.hdi(samples, hdi_prob=hdi_prob)
            fig.add_vline(x=mean_val, line_dash="dash", line_color="black",
                          row=1, col=i + 1)
            fig.add_vline(x=float(hdi_vals[0]), line_dash="dot",
                          line_color="firebrick", row=1, col=i + 1)
            fig.add_vline(x=float(hdi_vals[1]), line_dash="dot",
                          line_color="firebrick", row=1, col=i + 1)

        n = len(expanded)
        fig.update_layout(
            title_text=f"Posterior Distributions ({hdi_prob*100:.0f}% HDI)",
            width=figsize[0] if n < 4 else figsize[0] * n // 3,
            height=figsize[1], barmode='overlay')
        return fig

    # ------------------------------------------------------------------
    # autocor
    # ------------------------------------------------------------------
    def autocor_jax(self, m, var_names=None, exclude_vars=None,
                    filter_regex=None, max_lag=40):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np

        expanded = self._expand(m, var_names, exclude_vars, filter_regex)
        fig = make_subplots(
            rows=len(expanded), cols=1,
            subplot_titles=[f"Autocorrelation of {e[0]}" for e in expanded])

        for i, (name, samples) in enumerate(expanded):
            samples = np.asarray(samples)
            autocorr = [1.0] + [
                float(np.corrcoef(samples[:-t], samples[t:])[0, 1])
                for t in range(1, max_lag)
            ]
            color = _COLORS[i % len(_COLORS)]
            fig.add_trace(
                go.Bar(y=autocorr, name=name, marker_color=color,
                       showlegend=False),
                row=i + 1, col=1)

        fig.update_layout(height=250 * len(expanded),
                          title_text="Autocorrelation Plots",
                          barmode='group')
        return fig

    # ------------------------------------------------------------------
    # forest
    # ------------------------------------------------------------------
    def forest_jax(self, m, var_names=None, exclude_vars=None,
                   filter_regex=None, hdi_prob=0.95):
        import plotly.graph_objects as go
        import jax.numpy as jnp

        expanded = self._expand(m, var_names, exclude_vars, filter_regex)
        fig = go.Figure()

        for i, (name, samples) in enumerate(expanded):
            color = _COLORS[i % len(_COLORS)]
            fig.add_trace(go.Violin(
                x=samples, y=[f" {name} "], name=name, legendgroup=name,
                orientation='h', side='both', points=False, fillcolor=color,
                opacity=0.4, line_width=0, spanmode='hard'))

            mean_val = float(jnp.mean(samples))
            hdi_vals = jd.hdi(samples, hdi_prob=hdi_prob)
            lo, hi = float(hdi_vals[0]), float(hdi_vals[1])

            fig.add_trace(go.Scatter(
                x=[mean_val], y=[f" {name} "], mode='markers',
                legendgroup=name, name=name,
                marker=dict(color=color, size=8),
                error_x=dict(type='data', symmetric=False,
                             array=[hi - mean_val],
                             arrayminus=[mean_val - lo],
                             width=4, color=color),
                showlegend=False))

        fig.add_vline(x=0, line_dash="dash", line_color="black")
        fig.update_layout(
            title_text=f'Forest Plot ({hdi_prob * 100:.1f}% HDI)',
            xaxis_title="Parameter Value", yaxis_title="Parameter",
            violingap=0.1, plot_bgcolor='white')
        fig.update_yaxes(autorange="reversed")
        return fig

    # ------------------------------------------------------------------
    # density
    # ------------------------------------------------------------------
    def density_jax(self, m, var_names=None, exclude_vars=None,
                    filter_regex=None, shade=0.4):
        import plotly.graph_objects as go
        import plotly.colors as pcolors_
        from plotly.subplots import make_subplots
        import seaborn as sns
        import matplotlib.pyplot as plt

        expanded = self._expand(m, var_names, exclude_vars, filter_regex)
        fig = make_subplots(
            rows=len(expanded), cols=1,
            subplot_titles=[f"Density of {e[0]}" for e in expanded])

        for i, (name, samples) in enumerate(expanded):
            color = _COLORS[i % len(_COLORS)]
            rgb = pcolors_.hex_to_rgb(color)
            fill = f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{shade})'
            with sns.plotting_context(rc={"figure.figsize": (1, 1)}):
                kde_plot = sns.kdeplot(samples)
                kde = kde_plot.get_lines()[0].get_data()
                plt.close()
            fig.add_trace(
                go.Scatter(x=kde[0], y=kde[1], fill='tozeroy', mode='lines',
                           name=name, showlegend=False,
                           fillcolor=fill, line_color=color),
                row=i + 1, col=1)

        fig.update_layout(height=300 * len(expanded),
                          title_text="Density Plots")
        return fig

    # ------------------------------------------------------------------
    # pair
    # ------------------------------------------------------------------
    def pair_jax(self, m, var_names=None, exclude_vars=None, filter_regex=None,
                 colorscale="Viridis", max_points=1000,
                 point_color='rgba(40, 150, 200, 0.4)'):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd
        import numpy as np

        expanded = self._expand(m, var_names, exclude_vars, filter_regex)
        labels = [e[0] for e in expanded]
        samples_dict = {name: np.asarray(s) for (name, s) in expanded}
        n_vars = len(labels)

        df = pd.DataFrame(samples_dict)
        plot_df = (df.sample(n=max_points, random_state=42)
                   if len(df) > max_points else df)

        fig = make_subplots(rows=n_vars, cols=n_vars,
                            horizontal_spacing=0.03, vertical_spacing=0.03)

        for i in range(n_vars):
            for j in range(n_vars):
                v1, v2 = labels[i], labels[j]
                if i == j:
                    fig.add_trace(
                        go.Histogram(x=df[v1], marker_color='#440154'),
                        row=i + 1, col=j + 1)
                elif i > j:
                    fig.add_trace(
                        go.Histogram2dContour(
                            x=df[v2], y=df[v1], colorscale=colorscale,
                            showscale=False,
                            contours=dict(coloring='lines'),
                            line=dict(width=1)),
                        row=i + 1, col=j + 1)
                    fig.add_trace(
                        go.Scatter(x=plot_df[v2], y=plot_df[v1],
                                   mode='markers',
                                   marker=dict(size=3, color=point_color)),
                        row=i + 1, col=j + 1)
                    fig.add_trace(
                        go.Scatter(x=[df[v2].median()],
                                   y=[df[v1].median()], mode='markers',
                                   marker=dict(symbol='square',
                                               color='black', size=8)),
                        row=i + 1, col=j + 1)

        fig.update_layout(
            title_text="Pair Plot: Histograms, Density, and Samples",
            height=250 * n_vars, width=250 * n_vars,
            showlegend=False, plot_bgcolor='white')
        for i in range(n_vars):
            fig.update_yaxes(title_text=labels[i], row=i + 1, col=1,
                             showline=True, linewidth=1, linecolor='black',
                             mirror=True)
        for j in range(n_vars):
            fig.update_xaxes(title_text=labels[j], row=n_vars, col=j + 1,
                             showline=True, linewidth=1, linecolor='black',
                             mirror=True)
        return fig

    # ------------------------------------------------------------------
    # model_checks
    # ------------------------------------------------------------------
    def model_checks_jax(self, m, var_names=None, exclude_vars=None,
                         filter_regex=None):
        kw = dict(var_names=var_names, exclude_vars=exclude_vars,
                  filter_regex=filter_regex)
        print("Posterior Plots:")
        self.posterior(m, **kw).show()
        print("\nAutocorrelation Plots:")
        self.autocor(m, **kw).show()
        print("\nTrace Plots:")
        self.plot_trace(m, **kw).show()
        print("\nForest Plot:")
        self.forest(m, **kw).show()
        print("\nPair Plot:")
        self.pair(m, **kw).show()

    # ------------------------------------------------------------------
    # loo / waic / compute_log_likelihood
    # ------------------------------------------------------------------
    def compute_ll(self, m, model, *model_args, obs_name=None, **model_kwargs):
        """Compute pointwise log-likelihood by replaying the model."""
        return jd.compute_log_likelihood(
            model, _get_posteriors(m), *model_args,
            obs_name=obs_name, **model_kwargs)

    def loo_jax(self, m=None, model=None, *model_args, log_likelihood=None,
                obs_name=None, pointwise=False, scale="log", **model_kwargs):
        """PSIS-LOO-CV.

        Usage:
            d.loo(m, model, X, y)                          # auto-compute ll
            d.loo(log_likelihood=precomputed_array)        # direct
        """
        if log_likelihood is None:
            if m is None or model is None:
                raise ValueError(
                    "Either pass m + model + data args, or log_likelihood=.\n"
                    "  d.loo(m, model, X, y)\n"
                    "  d.loo(log_likelihood=precomputed_ll)")
            log_likelihood = jd.compute_log_likelihood(
                model, _get_posteriors(m), *model_args,
                obs_name=obs_name, **model_kwargs)
        return jd.loo(log_likelihood, pointwise=pointwise, scale=scale)

    def waic_jax(self, m=None, model=None, *model_args, log_likelihood=None,
                 obs_name=None, pointwise=False, scale="log", **model_kwargs):
        """WAIC.

        Usage:
            d.WAIC(m, model, X, y)
            d.WAIC(log_likelihood=precomputed_array)
        """
        if log_likelihood is None:
            if m is None or model is None:
                raise ValueError(
                    "Either pass m + model + data args, or log_likelihood=.\n"
                    "  d.WAIC(m, model, X, y)\n"
                    "  d.WAIC(log_likelihood=precomputed_ll)")
            log_likelihood = jd.compute_log_likelihood(
                model, _get_posteriors(m), *model_args,
                obs_name=obs_name, **model_kwargs)
        return jd.waic(log_likelihood, pointwise=pointwise, scale=scale)

    # ------------------------------------------------------------------
    # Apply all patches
    # ------------------------------------------------------------------
    cls.summary = summary_jax
    cls.rhat = rhat_jax
    cls.ess = ess_jax
    cls.mcse = mcse_jax
    cls.plot_trace = plot_trace_jax
    cls.posterior = posterior_jax
    cls.autocor = autocor_jax
    cls.forest = forest_jax
    cls.density = density_jax
    cls.pair = pair_jax
    cls.model_checks = model_checks_jax
    cls.loo = loo_jax
    cls.WAIC = waic_jax
    cls.compute_log_likelihood = compute_ll
    cls.compare = staticmethod(jd.compare)

    return cls