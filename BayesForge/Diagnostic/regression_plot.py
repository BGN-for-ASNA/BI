"""Posterior predictive regression overlay plots for BF models.

Produces scatter of observed data + N smooth posterior regression lines
+ posterior mean line, compatible with any GLM family.

Each individual line shows E[y|x, θ_s] (parameter uncertainty only, no
observation noise) computed by averaging n_smooth draws per posterior sample.
The mean line averages all posterior samples and is always smooth.

Usage via bind_diag_to_model:
    m.diag.plot_regression('weight', n=20)
    m.diag.plot_regression('x', link_inv=jax.nn.sigmoid, n=30)

Standalone:
    from BayesForge.Diagnostic.regression_plot import plot_regression
    fig = plot_regression(m, 'weight', n=20)
"""
import numpy as np
import jax
import jax.numpy as jnp
import plotly.graph_objects as go
import plotly.colors as pcolors
from numpyro.infer import Predictive

_COLORS = pcolors.qualitative.Plotly


def _find_obs_key(pred_dict, param_keys, obs_args):
    """Return the key in pred_dict corresponding to the observed outcome."""
    if obs_args and obs_args[0] in pred_dict:
        return obs_args[0]
    return next((k for k in pred_dict if k not in param_keys), list(pred_dict.keys())[0])


def _predict(model2, posterior_subset, pred_data, rng_key):
    """Run Predictive with given posterior subset dict."""
    return Predictive(model2, posterior_samples=posterior_subset)(rng_key, **pred_data)


def plot_regression(
    m,
    x_var,
    y_obs=None,
    n=20,
    link_inv=None,
    x_range=None,
    n_points=200,
    x_label=None,
    y_label=None,
    line_color="rgba(220,50,50,0.25)",
    mean_color="blue",
    seed=42,
    n_smooth=200,
):
    """Regression overlay: observed data + N smooth posterior lines + mean.

    Individual lines show E[y|x, θ_s] (parameter uncertainty only), computed
    by averaging n_smooth posterior predictive draws per selected sample so
    observation noise averages out. Works for any GLM family.

    Args:
        m: Fitted BF model instance.
        x_var: Name of the x predictor variable in m.data_on_model, or a
               1D numpy/jax array of x values.
        y_obs: Observed outcome array. Auto-detected from m.obs_args if None.
        n: Number of posterior regression lines to overlay.
        link_inv: Optional inverse link function applied to predictions
                  (e.g. jax.nn.sigmoid). If None no transform is applied.
        x_range: (min, max) tuple or 1D array for the x prediction grid.
                 Defaults to range of x_var data.
        n_points: Number of x grid points for prediction lines.
        x_label, y_label: Axis labels (defaults to variable names).
        line_color: Plotly RGBA string for individual lines.
        mean_color: Color for the posterior mean line.
        seed: RNG seed for reproducible line selection.
        n_smooth: Draws per selected posterior sample averaged to eliminate
                  observation noise from individual lines (default 200).

    Returns:
        plotly.graph_objects.Figure
    """
    # ------------------------------------------------------------------ data
    if isinstance(x_var, str):
        x_obs = np.asarray(m.data_on_model[x_var]).flatten()
        x_label = x_label or x_var
    else:
        x_obs = np.asarray(x_var).flatten()
        x_label = x_label or "x"

    if y_obs is None and m.obs_args:
        obs_name = m.obs_args[0]
        raw = m.data_on_model.get(obs_name)
        if raw is not None:
            y_obs = np.asarray(raw).flatten()
            y_label = y_label or obs_name

    # -------------------------------------------------------- x prediction grid
    if x_range is None:
        x_grid = np.linspace(float(np.min(x_obs)), float(np.max(x_obs)), n_points)
    else:
        x_range = np.asarray(x_range)
        x_grid = (np.linspace(float(x_range[0]), float(x_range[-1]), n_points)
                  if x_range.ndim == 1 and len(x_range) == 2 else x_range)
    x_grid_jax = jnp.array(x_grid)

    # -------------------------------------------- build pred_data (no obs args)
    if isinstance(x_var, str):
        pred_data = {k: v for k, v in m.data_on_model.items()
                     if k not in (m.obs_args or [])}
        pred_data[x_var] = x_grid_jax
    else:
        pred_data = {k: v for k, v in m.data_on_model.items()
                     if k not in (m.obs_args or [])}
        first_non_obs = next(
            (k for k in m.data_on_model if k not in (m.obs_args or [])), None
        )
        if first_non_obs:
            pred_data[first_non_obs] = x_grid_jax

    # ----------------------------------------- prepare model2 and posteriors
    m.build_model_with_Y_None(m.model)
    posteriors_flat = m.sampler.get_samples()           # {param: (S,)}
    S = next(iter(posteriors_flat.values())).shape[0]
    param_keys = set(posteriors_flat.keys())

    rng = np.random.default_rng(seed)
    idx = rng.choice(S, size=min(n, S), replace=False)
    n_sel = len(idx)

    # ---- find obs_key from a tiny 1-sample probe ----
    probe_post = {k: jnp.array(v)[:1] for k, v in posteriors_flat.items()}
    probe = _predict(m.model2, probe_post, pred_data, jax.random.PRNGKey(seed))
    obs_key = _find_obs_key(probe, param_keys, m.obs_args)
    n_pts = np.asarray(probe[obs_key]).shape[-1]          # actual output width

    # ---- individual lines: average n_smooth draws per parameter sample ----
    # Each draw ~ p(y|x,θ_s); averaging n_smooth ≈ E[y|x,θ_s] (smooth curve)
    sel_post = {k: jnp.repeat(jnp.array(v)[idx], n_smooth, axis=0)
                for k, v in posteriors_flat.items()}
    pred_sel = _predict(m.model2, sel_post, pred_data, jax.random.PRNGKey(seed + 1))
    yrep_raw = np.asarray(pred_sel[obs_key], dtype=np.float64)  # (n_sel*n_smooth, n_pts, ...)
    if yrep_raw.ndim > 2:
        yrep_raw = yrep_raw.reshape(yrep_raw.shape[0], -1)
    yrep_lines = yrep_raw.reshape(n_sel, n_smooth, -1).mean(axis=1)  # (n_sel, n_pts)

    # ---- mean line: average all S samples (LLN → smooth) ----
    all_post = {k: jnp.array(v) for k, v in posteriors_flat.items()}
    pred_all = _predict(m.model2, all_post, pred_data, jax.random.PRNGKey(seed + 2))
    yrep_all = np.asarray(pred_all[obs_key], dtype=np.float64)       # (S, n_pts, ...)
    if yrep_all.ndim > 2:
        yrep_all = yrep_all.reshape(yrep_all.shape[0], -1)
    mean_pred = yrep_all.mean(axis=0)                                  # (n_pts,)

    # align to x_grid length
    yrep_lines = yrep_lines[:, :len(x_grid)]
    mean_pred = mean_pred[:len(x_grid)]

    # --------------------------------------------------------- apply link inverse
    if link_inv is not None:
        yrep_lines = np.asarray(link_inv(jnp.array(yrep_lines)))
        mean_pred = np.asarray(link_inv(jnp.array(mean_pred)))
        y_plot = np.asarray(link_inv(jnp.array(y_obs))) if y_obs is not None else y_obs
    else:
        y_plot = y_obs

    # ---------------------------------------------------------------------- plot
    fig = go.Figure()

    for i in range(n_sel):
        fig.add_trace(go.Scatter(
            x=x_grid, y=yrep_lines[i],
            mode="lines",
            line=dict(color=line_color, width=0.8),
            showlegend=(i == 0),
            name=f"{n_sel} posterior lines",
        ))

    fig.add_trace(go.Scatter(
        x=x_grid, y=mean_pred,
        mode="lines",
        line=dict(color=mean_color, width=2.5),
        name="Posterior mean",
    ))

    if y_plot is not None and y_obs is not None:
        fig.add_trace(go.Scatter(
            x=np.asarray(x_obs), y=y_plot,
            mode="markers",
            marker=dict(color="black", size=7, opacity=0.75),
            name="Observed",
        ))

    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label or "y",
        title="Regression: Posterior Predictive Lines",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig
