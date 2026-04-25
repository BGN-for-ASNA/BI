"""
Pure JAX replacements for ArviZ diagnostic functions.

These operate directly on posterior samples shaped as (num_chains, num_samples, ...).
All functions are JIT-compilable for maximum performance.
"""

import jax
import jax.numpy as jnp
from functools import partial


# =============================================================================
# Posterior dict filtering
# =============================================================================

def _base_name(name: str) -> str:
    """Strip index suffix: 'var[0,1]' → 'var'."""
    return name.split('[')[0]


def filter_posterior_dict(posteriors: dict, include=None, exclude=None) -> dict:
    """Filter a posterior samples dict by parameter base name.

    Matching is on base name only: 'var' matches key 'var' (and covers all
    'var[i,j,...]') but never matches 'var_1'. Index suffixes in filter names
    are stripped, so 'var[0,0]' and 'var' are equivalent filters.

    Args:
        posteriors: Dict of {name: array}.
        include: str or list of str — keep only these base names.
        exclude: str or list of str — remove these base names.

    Returns:
        Filtered dict (shallow copy of matched entries).
    """
    def _as_set(arg):
        if arg is None:
            return None
        if isinstance(arg, str):
            arg = [arg]
        return {_base_name(n) for n in arg}

    inc = _as_set(include)
    exc = _as_set(exclude)

    keys = [k for k in posteriors
            if (inc is None or k in inc) and (exc is None or k not in exc)]
    return {k: posteriors[k] for k in keys}


# =============================================================================
# R-hat (Split R-hat, Vehtari et al. 2021)
# =============================================================================

@partial(jax.jit, static_argnames=())
def _rhat_1d(chains: jnp.ndarray) -> jnp.ndarray:
    """Compute split R-hat for a single parameter.

    Args:
        chains: Array of shape (num_chains, num_samples).

    Returns:
        Scalar R-hat value.
    """
    num_chains, num_samples = chains.shape

    # Split each chain in half -> 2 * num_chains half-chains
    half = num_samples // 2
    first_half = chains[:, :half]
    second_half = chains[:, half:2 * half]
    split_chains = jnp.concatenate([first_half, second_half], axis=0)  # (2*C, half)

    m = split_chains.shape[0]  # number of split chains
    n = split_half = half       # length of each split chain

    # Chain means and overall mean
    chain_means = jnp.mean(split_chains, axis=1)  # (m,)
    overall_mean = jnp.mean(chain_means)

    # Between-chain variance B
    B = n / (m - 1) * jnp.sum((chain_means - overall_mean) ** 2)

    # Within-chain variance W
    chain_vars = jnp.var(split_chains, axis=1, ddof=1)  # (m,)
    W = jnp.mean(chain_vars)

    # Pooled variance estimate
    var_hat = (n - 1) / n * W + B / n

    # R-hat
    rhat = jnp.sqrt(var_hat / W)
    return rhat


def rhat(posterior_samples: dict, var_names=None,
         include=None, exclude=None) -> dict:
    """Compute R-hat for all parameters.

    Args:
        posterior_samples: Dict of {name: array}, shape (num_chains, num_samples, ...).
        include: str or list — keep only these base names.
        exclude: str or list — remove these base names.
        var_names: legacy alias for include.

    Returns:
        Dict of {name: R-hat value(s)}.
    """
    posterior_samples = filter_posterior_dict(
        posterior_samples, include=include or var_names, exclude=exclude)
    var_names = list(posterior_samples.keys())

    results = {}
    for name in var_names:
        samples = posterior_samples[name]
        if samples.ndim == 2:
            # Scalar parameter: (chains, samples)
            results[name] = float(_rhat_1d(samples))
        else:
            # Multi-dimensional parameter: (chains, samples, ...)
            param_shape = samples.shape[2:]
            rhat_vals = jnp.empty(param_shape)
            import numpy as np
            for idx in np.ndindex(param_shape):
                chain_slice = samples[(slice(None), slice(None)) + idx]  # (C, S)
                rhat_vals = rhat_vals.at[idx].set(_rhat_1d(chain_slice))
            results[name] = rhat_vals
    return results


# =============================================================================
# Effective Sample Size (ESS) — bulk and tail
# =============================================================================

@partial(jax.jit, static_argnames=('max_lag',))
def _autocorr_1d(x: jnp.ndarray, max_lag: int = None) -> jnp.ndarray:
    """Compute autocorrelation using FFT for a single 1D chain."""
    n = x.shape[0]
    if max_lag is None:
        max_lag = n
    x_centered = x - jnp.mean(x)
    import numpy as _np
    fft_size = 2 ** int(_np.ceil(_np.log2(2 * n - 1)))
    fft_x = jnp.fft.rfft(x_centered, n=fft_size)
    acf = jnp.fft.irfft(fft_x * jnp.conj(fft_x), n=fft_size)[:n]
    acf = acf / acf[0]
    return acf[:max_lag]


def _autocov_np(ary):
    """Autocovariance for 2D array (n_chain, n_draw) at all lags, matching ArviZ."""
    import numpy as _np
    ary = _np.asarray(ary, dtype=float)
    n = ary.shape[1]
    from scipy.fft import next_fast_len
    m = next_fast_len(2 * n)
    ary = ary - ary.mean(axis=1, keepdims=True)
    fft_x = _np.fft.rfft(ary, n=m, axis=1)
    fft_x *= _np.conj(fft_x)
    acov = _np.fft.irfft(fft_x, n=m, axis=1)[:, :n]
    acov /= n
    return acov


def _split_chains_np(ary):
    """Split each chain in half and stack (ArviZ convention)."""
    import numpy as _np
    ary = _np.asarray(ary)
    half = ary.shape[1] // 2
    return _np.concatenate([ary[:, :half], ary[:, -half:]], axis=0)


def _z_scale_np(ary):
    """Rank-normalize array using Blom (1958) formula, matching ArviZ _z_scale."""
    import numpy as _np
    from scipy import stats as _stats
    ary = _np.asarray(ary, dtype=float)
    shape = ary.shape
    flat = ary.flatten()
    rank = _stats.rankdata(flat, method="average")
    c = 3.0 / 8.0
    n = len(rank)
    rank = (rank - c) / (n - 2.0 * c + 1.0)
    z = _stats.norm.ppf(rank)
    return z.reshape(shape)


def _ess_raw(ary):
    """Core ESS on pre-split, pre-transformed chains.

    Matches ArviZ _ess exactly: cross-chain pooled rho_hat, Geyer's initial
    positive sequence + initial monotone sequence, tau lower bound.

    Args:
        ary: (n_chain, n_draw) numpy array, already split and transformed.

    Returns:
        Scalar ESS float.
    """
    import numpy as _np
    ary = _np.asarray(ary, dtype=float)
    n_chain, n_draw = ary.shape

    acov = _autocov_np(ary)
    chain_mean = ary.mean(axis=1)
    mean_var = _np.mean(acov[:, 0]) * n_draw / (n_draw - 1.0)
    var_plus = mean_var * (n_draw - 1.0) / n_draw
    if n_chain > 1:
        var_plus += _np.var(chain_mean, ddof=1)

    rho_hat_t = _np.zeros(n_draw)
    rho_hat_even = 1.0
    rho_hat_t[0] = rho_hat_even
    rho_hat_odd = 1.0 - (mean_var - _np.mean(acov[:, 1])) / var_plus
    rho_hat_t[1] = rho_hat_odd

    # Geyer's initial positive sequence
    t = 1
    while t < (n_draw - 3) and (rho_hat_even + rho_hat_odd) > 0.0:
        rho_hat_even = 1.0 - (mean_var - _np.mean(acov[:, t + 1])) / var_plus
        rho_hat_odd = 1.0 - (mean_var - _np.mean(acov[:, t + 2])) / var_plus
        if (rho_hat_even + rho_hat_odd) >= 0:
            rho_hat_t[t + 1] = rho_hat_even
            rho_hat_t[t + 2] = rho_hat_odd
        t += 2

    max_t = t - 2
    if rho_hat_even > 0:
        rho_hat_t[max_t + 1] = rho_hat_even

    # Geyer's initial monotone sequence
    t = 1
    while t <= max_t - 2:
        if (rho_hat_t[t + 1] + rho_hat_t[t + 2]) > (rho_hat_t[t - 1] + rho_hat_t[t]):
            rho_hat_t[t + 1] = (rho_hat_t[t - 1] + rho_hat_t[t]) / 2.0
            rho_hat_t[t + 2] = rho_hat_t[t + 1]
        t += 2

    ess_total = float(n_chain * n_draw)
    tau_hat = -1.0 + 2.0 * _np.sum(rho_hat_t[:max_t + 1]) + _np.sum(rho_hat_t[max_t + 1:max_t + 2])
    tau_hat = max(tau_hat, 1.0 / _np.log10(ess_total))
    return ess_total / tau_hat


def _ess_1d(chains) -> float:
    """Bulk ESS: split chains → rank-normalize → _ess_raw (matches ArviZ _ess_bulk)."""
    import numpy as _np
    chains = _np.asarray(chains, dtype=float)
    split = _split_chains_np(chains)
    z = _z_scale_np(split)
    return float(_ess_raw(z))


def _ess_tail_1d(chains) -> float:
    """Tail ESS: min(ESS(I(x<=q05)), ESS(I(x<=q95))) (matches ArviZ _ess_tail)."""
    import numpy as _np
    chains = _np.asarray(chains, dtype=float)
    flat = chains.flatten()
    q05 = _np.percentile(flat, 5)
    q95 = _np.percentile(flat, 95)
    ess_low = _ess_raw(_split_chains_np((chains <= q05).astype(float)))
    ess_high = _ess_raw(_split_chains_np((chains <= q95).astype(float)))
    return float(min(ess_low, ess_high))


def ess(posterior_samples: dict, var_names=None, kind="bulk",
        include=None, exclude=None) -> dict:
    """Compute effective sample size for all parameters.

    Args:
        posterior_samples: Dict of {name: array}, shape (chains, samples, ...).
        include: str or list — keep only these base names.
        exclude: str or list — remove these base names.
        var_names: legacy alias for include.
        kind: "bulk" or "tail".

    Returns:
        Dict of {name: ESS value(s)}.
    """
    posterior_samples = filter_posterior_dict(
        posterior_samples, include=include or var_names, exclude=exclude)

    ess_fn = _ess_1d if kind == "bulk" else _ess_tail_1d

    results = {}
    for name, samples in posterior_samples.items():
        if samples.ndim == 2:
            results[name] = ess_fn(samples)
        else:
            param_shape = samples.shape[2:]
            import numpy as np
            ess_vals = np.empty(param_shape)
            for idx in np.ndindex(param_shape):
                chain_slice = samples[(slice(None), slice(None)) + idx]
                ess_vals[idx] = ess_fn(chain_slice)
            results[name] = ess_vals
    return results


# =============================================================================
# HDI (Highest Density Interval)
# =============================================================================

def hdi(samples: jnp.ndarray, hdi_prob: float = 0.94) -> jnp.ndarray:
    """Compute the Highest Density Interval (narrowest credible interval).

    Args:
        samples: 1D array of posterior samples.
        hdi_prob: Probability mass of the HDI (default 0.94).

    Returns:
        Array of [lower, upper] bounds.
    """
    samples = jnp.sort(samples.flatten())
    n = samples.shape[0]
    interval_size = int(jnp.ceil(hdi_prob * n))

    # Slide a window of size interval_size and find the narrowest
    if interval_size >= n:
        return jnp.array([samples[0], samples[-1]])

    # Width of every possible interval
    starts = samples[:n - interval_size]
    ends = samples[interval_size:]
    widths = ends - starts

    best_idx = int(jnp.argmin(widths))
    return jnp.array([samples[best_idx], samples[best_idx + interval_size]])


# =============================================================================
# Summary table (replaces az-dependent summary)
# =============================================================================

def summary(posterior_samples: dict, round_to: int = 2, hdi_prob: float = 0.89,
            var_names=None, exclude_vars=None, filter_regex=None,
            include=None, exclude=None,
            group_by_chain: bool = False):
    """Compute summary statistics matching az.summary() output format.

    Args:
        posterior_samples: Dict of {name: array}.
            When group_by_chain=False (default): shape (n_samples, ...) — flat.
            When group_by_chain=True: shape (chains, samples, ...).
        round_to: Decimal places to round to.
        hdi_prob: HDI probability mass (default 0.89 matches az.summary default).
        var_names: Optional list of variable names to include.
        exclude_vars: Optional list of variable names to exclude.
        filter_regex: Optional regex string to filter variable names.
        group_by_chain: If True, input is (chains, samples, ...). Default False.

    Returns:
        pandas DataFrame matching az.summary() columns:
        mean, sd, hdi_5.5%, hdi_94.5%, mcse_mean, mcse_sd, ess_bulk, ess_tail, r_hat
    """
    import pandas as pd
    import numpy as np
    import re as _re

    # include/exclude take priority; var_names/exclude_vars are legacy aliases
    _inc = include if include is not None else var_names
    _exc = exclude if exclude is not None else exclude_vars
    posterior_samples = filter_posterior_dict(posterior_samples,
                                              include=_inc, exclude=_exc)

    vars_to_process = list(posterior_samples.keys())

    if filter_regex is not None:
        vars_to_process = [v for v in vars_to_process if _re.search(filter_regex, v)]

    # Column names match ArviZ convention: hdi_{lo:.1f}% / hdi_{hi:.1f}%
    lo_pct = (1 - hdi_prob) / 2 * 100   # 5.5 for hdi_prob=0.89
    hi_pct = 100 - lo_pct                # 94.5
    hdi_lo_col = f'hdi_{lo_pct:.1f}%'
    hdi_hi_col = f'hdi_{hi_pct:.1f}%'

    summary_stats = {}

    for var_name in vars_to_process:
        samples = posterior_samples[var_name]

        if not group_by_chain:
            # Flat input (n_samples, ...) → (1, n_samples, ...)
            samples = jnp.expand_dims(samples, 0)

        # samples is now (C, S, ...)
        num_chains = samples.shape[0]
        param_shape = samples.shape[2:]

        if not param_shape:
            # Scalar parameter
            all_samples = samples.flatten()
            chains_2d = samples  # (C, S)

            mean_val = float(jnp.mean(all_samples))
            sd_val = float(jnp.std(all_samples))
            hdi_vals = hdi(all_samples, hdi_prob=hdi_prob)
            rhat_val = float(_rhat_1d(chains_2d)) if num_chains > 1 else float('nan')
            ess_val = _ess_1d(chains_2d)
            ess_tail_val = _ess_tail_1d(chains_2d)
            mcse_mean_val = sd_val / (ess_val ** 0.5) if ess_val > 0 else float('nan')
            mcse_sd_val = (sd_val / ((2 * (ess_val - 1)) ** 0.5)
                          if ess_val > 1 else float('nan'))

            summary_stats[var_name] = {
                'mean': mean_val, 'sd': sd_val,
                hdi_lo_col: float(hdi_vals[0]),
                hdi_hi_col: float(hdi_vals[1]),
                'mcse_mean': mcse_mean_val,
                'mcse_sd': mcse_sd_val,
                'ess_bulk': ess_val,
                'ess_tail': ess_tail_val,
                'r_hat': rhat_val,
            }
        else:
            for idx in np.ndindex(param_shape):
                idx_str = "[" + ", ".join(map(str, idx)) + "]"
                full_name = f"{var_name}{idx_str}"
                element_samples = samples[(slice(None), slice(None)) + idx]
                all_samples = element_samples.flatten()

                mean_val = float(jnp.mean(all_samples))
                sd_val = float(jnp.std(all_samples))
                hdi_vals = hdi(all_samples, hdi_prob=hdi_prob)
                rhat_val = float(_rhat_1d(element_samples)) if num_chains > 1 else float('nan')
                ess_val = _ess_1d(element_samples)
                ess_tail_val = _ess_tail_1d(element_samples)
                mcse_mean_val = sd_val / (ess_val ** 0.5) if ess_val > 0 else float('nan')
                mcse_sd_val = (sd_val / ((2 * (ess_val - 1)) ** 0.5)
                              if ess_val > 1 else float('nan'))

                summary_stats[full_name] = {
                    'mean': mean_val, 'sd': sd_val,
                    hdi_lo_col: float(hdi_vals[0]),
                    hdi_hi_col: float(hdi_vals[1]),
                    'mcse_mean': mcse_mean_val,
                    'mcse_sd': mcse_sd_val,
                    'ess_bulk': ess_val,
                    'ess_tail': ess_tail_val,
                    'r_hat': rhat_val,
                }

    return pd.DataFrame(summary_stats).T.round(round_to)


# =============================================================================
# MCSE (Monte Carlo Standard Error)
# =============================================================================

def mcse(posterior_samples: dict, var_names=None,
         include=None, exclude=None) -> dict:
    """Compute Monte Carlo Standard Error for all parameters.

    MCSE = posterior_std / sqrt(ESS_bulk)

    Args:
        posterior_samples: Dict of {name: array}, shape (chains, samples, ...).
        include: str or list — keep only these base names.
        exclude: str or list — remove these base names.
        var_names: legacy alias for include.

    Returns:
        Dict of {name: MCSE value(s)}.
    """
    posterior_samples = filter_posterior_dict(
        posterior_samples, include=include or var_names, exclude=exclude)

    ess_results = ess(posterior_samples, kind="bulk")

    results = {}
    for name, samples in posterior_samples.items():
        std_val = float(jnp.std(samples.flatten()))
        ess_val = ess_results[name]
        if isinstance(ess_val, (int, float)):
            results[name] = std_val / (ess_val ** 0.5)
        else:
            import numpy as np
            results[name] = np.array(
                [float(jnp.std(samples[(slice(None), slice(None)) + idx].flatten()))
                 / (ess_val[idx] ** 0.5)
                 for idx in np.ndindex(ess_val.shape)]
            ).reshape(ess_val.shape)
    return results


# =============================================================================
# PSIS (Pareto Smoothed Importance Sampling)
# Reference: Vehtari, Gelman, Gabry (2017). arxiv.org/abs/1507.02646
# =============================================================================

def _psis_smooth_tail(log_ratios_tail: jnp.ndarray) -> tuple:
    """Fit a Generalized Pareto Distribution (GPD) to the upper tail of
    importance ratios and replace the tail with order statistics from
    the fitted GPD.

    This is the core of PSIS: instead of using raw importance weights
    (which can have infinite variance), we fit a GPD to the largest
    weights and replace them with expected order statistics from that
    distribution — stabilizing the estimate.

    Args:
        log_ratios_tail: Sorted log importance ratios for the tail
            (the largest M values, where M ~ min(S/5, 3*sqrt(S))).

    Returns:
        (smoothed_log_ratios, pareto_k): The smoothed tail log-ratios
            and the estimated Pareto shape parameter k.
    """
    M = log_ratios_tail.shape[0]
    if M < 5:
        # Too few tail samples to fit — return raw values with bad k
        return log_ratios_tail, jnp.inf

    # Shift so the minimum tail value is 0 (work with exceedances)
    cutoff = log_ratios_tail[0]
    exceedances = log_ratios_tail - cutoff

    # Fit GPD via the Zhang & Stephens (2009) estimator
    # This is the method used by ArviZ / loo / Stan
    k_hat, sigma_hat = _fit_gpd(jnp.exp(exceedances))

    if jnp.isfinite(k_hat):
        # Compute expected order statistics of the fitted GPD
        # p_i = (i - 0.5) / M for i = 1..M
        p = (jnp.arange(0.5, M) / M)
        # GPD quantile function: sigma/k * ((1-p)^(-k) - 1) when k != 0
        #                        -sigma * log(1-p) when k == 0
        smoothed = jnp.where(
            jnp.abs(k_hat) > 1e-6,
            sigma_hat / k_hat * ((1 - p) ** (-k_hat) - 1),
            -sigma_hat * jnp.log(1 - p)
        )
        smoothed_log = jnp.log(smoothed + 1e-30) + cutoff
        # Ensure the smoothed tail doesn't exceed the max observed
        smoothed_log = jnp.minimum(smoothed_log, log_ratios_tail[-1])
    else:
        smoothed_log = log_ratios_tail
        k_hat = jnp.inf

    return smoothed_log, k_hat


def _fit_gpd(x: jnp.ndarray) -> tuple:
    """Fit Generalized Pareto Distribution using the empirical Bayes method
    of Zhang & Stephens (2009), as implemented in the R `loo` package.

    Args:
        x: 1D array of positive exceedances (already exponentiated).

    Returns:
        (k, sigma): Shape and scale parameters of the GPD.
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    if N < 5:
        return jnp.inf, jnp.nan

    x_sorted = np.sort(x)
    # Remove any zeros or negatives
    x_sorted = x_sorted[x_sorted > 0]
    N = len(x_sorted)
    if N < 5:
        return jnp.inf, jnp.nan

    # Zhang & Stephens (2009) prior on theta = 1/sigma
    m = 20 + int(np.sqrt(N))  # number of grid points
    x_mean = np.mean(x_sorted)

    # Grid of theta values (prior quartiles)
    # theta_i = 1/x_{ceil(N*p_i)} where p_i spans from near 0 to near 1
    prior = 3.0
    b_vec = np.array([1.0 / x_sorted[int(np.ceil(N * (i + 0.5) / (m))) - 1]
                       if int(np.ceil(N * (i + 0.5) / m)) <= N
                       else 1.0 / x_sorted[-1]
                       for i in range(m)])
    # Clamp to [0, 2*mean based bound]
    b_vec = np.clip(b_vec, 1e-10, 2.0 / max(x_mean, 1e-10))

    # Log-likelihood profile for each theta
    # For GPD with shape k and scale sigma, if we parametrize theta = (1+k)/sigma:
    # The profile log-likelihood for theta is:
    # l(theta) = N*(log(theta/(1+k)) - (1+1/k)*mean(log(1+k*x/sigma)))
    # Using the moment-based approach:
    k_vec = np.zeros(m)
    sigma_vec = np.zeros(m)
    log_lik = np.zeros(m)

    for i in range(m):
        theta = b_vec[i]
        # k = mean(log(1 - theta*x)) with negative sign adjustments
        temp = 1.0 - theta * x_sorted
        temp = np.clip(temp, 1e-30, None)
        k_est = np.mean(np.log(temp))
        # sigma = -k/theta
        k_vec[i] = -k_est
        sigma_vec[i] = -k_est / max(theta, 1e-30)
        # log-likelihood
        if k_vec[i] > -0.5:
            log_lik[i] = N * (np.log(theta) + k_est - 1.0)
        else:
            log_lik[i] = -np.inf

    # Posterior weights (with uniform prior)
    log_lik_max = np.max(log_lik)
    weights = np.exp(log_lik - log_lik_max)
    weights = weights / np.sum(weights)

    # Posterior mean of k and sigma
    k_hat = float(np.sum(weights * k_vec))
    sigma_hat = float(np.sum(weights * sigma_vec))

    return jnp.array(k_hat), jnp.array(sigma_hat)


def _psis_weights(log_likelihood_i: jnp.ndarray) -> tuple:
    """Compute PSIS-smoothed log weights for a single data point.

    The importance ratios for LOO are r_s = 1/p(y_i | theta_s),
    so log_ratios = -log_likelihood for that data point.

    Args:
        log_likelihood_i: Array of shape (S,) — log p(y_i | theta_s)
            for all S posterior draws (chains already flattened).

    Returns:
        (log_weights, pareto_k): Smoothed log importance weights
            and the Pareto k diagnostic.
    """
    S = log_likelihood_i.shape[0]

    # Log importance ratios (negate log-lik to get log(1/p(y|theta)))
    log_ratios = -log_likelihood_i

    # Stabilize by subtracting max
    log_ratios_max = jnp.max(log_ratios)
    log_ratios_centered = log_ratios - log_ratios_max

    # Determine tail cutoff: M = min(S/5, 3*sqrt(S))
    M = int(min(S / 5, 3 * (S ** 0.5)))
    M = max(M, 5)

    # Sort and identify the tail
    sorted_indices = jnp.argsort(log_ratios_centered)
    sorted_log_ratios = log_ratios_centered[sorted_indices]

    # Smooth only the upper tail (largest M values)
    tail_start = S - M
    tail = sorted_log_ratios[tail_start:]
    smoothed_tail, pareto_k = _psis_smooth_tail(tail)

    # Reconstruct: keep body, replace tail
    smoothed_sorted = jnp.concatenate([sorted_log_ratios[:tail_start], smoothed_tail])

    # Un-sort back to original order
    unsorted_indices = jnp.argsort(sorted_indices)
    log_weights = smoothed_sorted[unsorted_indices] + log_ratios_max

    # Normalize (in log space)
    log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)

    return log_weights_normalized, pareto_k


# =============================================================================
# LOO-CV (PSIS-LOO)
# =============================================================================

class ELPDData:
    """Container for ELPD results, similar to ArviZ ELPDData."""

    def __init__(self, kind, elpd, se, p, n_samples, n_data_points,
                 pointwise_elpd, pareto_k, scale, warning):
        self.kind = kind
        if kind == "loo":
            self.elpd_loo = elpd
            self.p_loo = p
        else:
            self.elpd_waic = elpd
            self.p_waic = p
        self.elpd = elpd
        self.se = se
        self.p = p
        self.n_samples = n_samples
        self.n_data_points = n_data_points
        self.pointwise_elpd = pointwise_elpd
        self.pareto_k = pareto_k
        self.scale = scale
        self.warning = warning
        self.good_k = min(1 - 1 / max(jnp.log10(n_samples), 1.01), 0.7) if n_samples > 10 else 0.7

    def __repr__(self):
        kind_upper = self.kind.upper()
        lines = [
            f"Computed from {self.n_samples} posterior samples and "
            f"{self.n_data_points} observations (log scale).",
            "",
            f"         Estimate    SE",
            f"elpd_{self.kind:4s}  {self.elpd:10.2f}  {self.se:.2f}",
            f"p_{self.kind:4s}     {self.p:10.2f}        -",
            "",
        ]
        if self.pareto_k is not None:
            n_bad = int(jnp.sum(jnp.array(self.pareto_k) > self.good_k))
            if n_bad > 0:
                lines.append(
                    f"WARNING: {n_bad} Pareto k values > {self.good_k:.2f}. "
                    f"LOO estimate may be unreliable."
                )
            else:
                lines.append(
                    f"All Pareto k estimates OK (k < {self.good_k:.2f})."
                )
        return "\n".join(lines)


def loo(log_likelihood: jnp.ndarray, pointwise: bool = False,
        scale: str = "log") -> ELPDData:
    """Compute PSIS-LOO-CV from pointwise log-likelihood values.

    Implements Pareto-smoothed importance sampling leave-one-out
    cross-validation (Vehtari et al., 2017).

    Args:
        log_likelihood: Array of shape (chains, samples, n_data_points)
            or (total_samples, n_data_points) containing pointwise
            log-likelihood values log p(y_i | theta_s).
        pointwise: If True, include pointwise values in the result.
        scale: "log" (default), "negative_log", or "deviance".

    Returns:
        ELPDData object with elpd_loo, p_loo, SE, etc.
    """
    # Flatten chains if needed: (C, S, N) -> (C*S, N)
    if log_likelihood.ndim == 3:
        C, S, N = log_likelihood.shape
        ll_flat = log_likelihood.reshape(C * S, N)
    elif log_likelihood.ndim == 2:
        ll_flat = log_likelihood
    else:
        raise ValueError(
            f"log_likelihood must be 2D or 3D, got shape {log_likelihood.shape}")

    total_S, N = ll_flat.shape

    # Compute PSIS weights and LOO-elpd for each data point
    loo_lppd_i = jnp.zeros(N)
    pareto_k = jnp.zeros(N)

    for i in range(N):
        log_w, k_i = _psis_weights(ll_flat[:, i])
        # LOO log predictive density for point i:
        # log( sum_s w_s * p(y_i | theta_s) ) where w_s are normalized PSIS weights
        # = logsumexp(log_w + log_lik_i)
        loo_lppd_i = loo_lppd_i.at[i].set(
            jax.scipy.special.logsumexp(log_w + ll_flat[:, i])
        )
        pareto_k = pareto_k.at[i].set(k_i)

    # elpd_loo = sum of pointwise loo_lppd
    elpd_loo = jnp.sum(loo_lppd_i)

    # p_loo (effective number of parameters)
    # = lppd - elpd_loo, where lppd = sum_i log(mean_s p(y_i|theta_s))
    lppd_i = jax.scipy.special.logsumexp(ll_flat, axis=0) - jnp.log(total_S)
    lppd = jnp.sum(lppd_i)
    p_loo = lppd - elpd_loo

    # Standard error
    se = jnp.sqrt(N * jnp.var(loo_lppd_i))

    # Warning
    good_k = min(1 - 1 / max(float(jnp.log10(total_S)), 1.01), 0.7)
    has_warning = bool(jnp.any(pareto_k > good_k))

    # Scale
    elpd_out = float(elpd_loo)
    if scale == "negative_log":
        elpd_out = -elpd_out
    elif scale == "deviance":
        elpd_out = -2 * elpd_out

    return ELPDData(
        kind="loo",
        elpd=elpd_out,
        se=float(se),
        p=float(p_loo),
        n_samples=int(total_S),
        n_data_points=int(N),
        pointwise_elpd=loo_lppd_i if pointwise else None,
        pareto_k=pareto_k if pointwise else pareto_k,
        scale=scale,
        warning=has_warning,
    )


# =============================================================================
# WAIC (Widely Applicable Information Criterion)
# =============================================================================

def waic(log_likelihood: jnp.ndarray, pointwise: bool = False,
         scale: str = "log") -> ELPDData:
    """Compute WAIC from pointwise log-likelihood values.

    The Widely Applicable Information Criterion (Watanabe, 2010)
    estimates the out-of-sample predictive accuracy using the
    computed log pointwise predictive density and a correction
    for the effective number of parameters.

    Args:
        log_likelihood: Array of shape (chains, samples, n_data_points)
            or (total_samples, n_data_points) containing pointwise
            log-likelihood values log p(y_i | theta_s).
        pointwise: If True, include pointwise values in result.
        scale: "log" (default), "negative_log", or "deviance".

    Returns:
        ELPDData object with elpd_waic, p_waic, SE, etc.
    """
    # Flatten chains if needed
    if log_likelihood.ndim == 3:
        C, S, N = log_likelihood.shape
        ll_flat = log_likelihood.reshape(C * S, N)
    elif log_likelihood.ndim == 2:
        ll_flat = log_likelihood
    else:
        raise ValueError(
            f"log_likelihood must be 2D or 3D, got shape {log_likelihood.shape}")

    total_S, N = ll_flat.shape

    # lppd = sum_i log( mean_s( p(y_i | theta_s) ) )
    # In log space: logsumexp over samples - log(S)
    lppd_i = jax.scipy.special.logsumexp(ll_flat, axis=0) - jnp.log(total_S)

    # p_waic = sum_i var_s( log p(y_i | theta_s) )
    # Variance of log-lik across posterior samples, for each data point
    p_waic_i = jnp.var(ll_flat, axis=0)

    # elpd_waic = lppd - p_waic (pointwise)
    elpd_waic_i = lppd_i - p_waic_i
    elpd_waic = jnp.sum(elpd_waic_i)
    p_waic = jnp.sum(p_waic_i)

    # Standard error
    se = jnp.sqrt(N * jnp.var(elpd_waic_i))

    # Warning: if any p_waic_i > 0.4
    has_warning = bool(jnp.any(p_waic_i > 0.4))

    # Scale
    elpd_out = float(elpd_waic)
    if scale == "negative_log":
        elpd_out = -elpd_out
    elif scale == "deviance":
        elpd_out = -2 * elpd_out

    return ELPDData(
        kind="waic",
        elpd=elpd_out,
        se=float(se),
        p=float(p_waic),
        n_samples=int(total_S),
        n_data_points=int(N),
        pointwise_elpd=elpd_waic_i if pointwise else None,
        pareto_k=None,
        scale=scale,
        warning=has_warning,
    )


# =============================================================================
# Model comparison (compare)
# =============================================================================

def compare(compare_dict: dict, ic: str = "loo", method: str = "stacking",
            scale: str = "log") -> "pd.DataFrame":
    """Compare models based on ELPD (LOO or WAIC).

    Args:
        compare_dict: Dict of {model_name: log_likelihood_array}.
            Each array has shape (chains, samples, n_data) or
            (total_samples, n_data).
        ic: "loo" or "waic".
        method: Weight estimation method — "stacking" (recommended)
            or "pseudo-BMA".
        scale: "log", "negative_log", or "deviance".

    Returns:
        pandas DataFrame ordered from best to worst model with columns:
        rank, elpd, pIC, elpd_diff, weight, SE, dSE, warning.
    """
    import pandas as pd
    import numpy as np

    ic_fn = loo if ic == "loo" else waic

    # Compute IC for all models
    results = {}
    pointwise_elpds = {}
    for name, ll in compare_dict.items():
        res = ic_fn(ll, pointwise=True, scale="log")  # always compute in log first
        results[name] = res
        pointwise_elpds[name] = np.array(res.pointwise_elpd)

    model_names = list(results.keys())
    K = len(model_names)
    N = results[model_names[0]].n_data_points

    # Stack pointwise elpds: (K, N)
    elpd_matrix = np.stack([pointwise_elpds[m] for m in model_names])

    # Compute weights
    if method == "stacking":
        weights = _stacking_weights(elpd_matrix)
    elif method == "pseudo-BMA":
        weights = _pseudo_bma_weights(elpd_matrix)
    else:
        weights = np.ones(K) / K

    # Build comparison table
    rows = []
    elpd_values = np.array([float(results[m].elpd) for m in model_names])
    ranks = np.argsort(-elpd_values)  # higher elpd = better

    # Reference is the best model
    best_idx = ranks[0]
    best_elpd_pointwise = elpd_matrix[best_idx]

    for idx, name in enumerate(model_names):
        res = results[name]
        diff_pointwise = elpd_matrix[idx] - best_elpd_pointwise
        elpd_diff = np.sum(diff_pointwise)
        dse = np.sqrt(N * np.var(diff_pointwise)) if idx != best_idx else 0.0

        rows.append({
            'rank': int(np.where(ranks == idx)[0][0]),
            f'elpd_{ic}': float(res.elpd),
            f'p_{ic}': float(res.p),
            'elpd_diff': float(elpd_diff),
            'weight': float(weights[idx]),
            'SE': float(res.se),
            'dSE': float(dse),
            'warning': res.warning,
        })

    df = pd.DataFrame(rows, index=model_names)
    df = df.sort_values('rank')

    # Apply scale
    if scale != "log":
        multiplier = -1.0 if scale == "negative_log" else -2.0
        for col in [f'elpd_{ic}', 'elpd_diff']:
            df[col] = df[col] * multiplier

    return df


def _stacking_weights(elpd_matrix: "np.ndarray") -> "np.ndarray":
    """Compute stacking weights by optimizing the combined LOO predictive
    density (Yao et al., 2018).

    This solves:  max_w  sum_i log( sum_k w_k * exp(elpd_k_i) )
                  s.t.   w >= 0, sum(w) = 1

    Uses scipy's minimize with SLSQP.

    Args:
        elpd_matrix: (K, N) array of pointwise elpds.

    Returns:
        Array of shape (K,) with stacking weights.
    """
    import numpy as np
    from scipy.optimize import minimize

    K, N = elpd_matrix.shape

    if K == 1:
        return np.array([1.0])

    # Normalize per data point for numerical stability
    elpd_max = elpd_matrix.max(axis=0, keepdims=True)
    exp_elpd = np.exp(elpd_matrix - elpd_max)  # (K, N)

    def neg_log_score(w):
        # Weighted combination: for each data point i, compute sum_k w_k * exp_elpd_ki
        combined = w @ exp_elpd  # (N,)
        combined = np.clip(combined, 1e-30, None)
        return -np.sum(np.log(combined))

    def neg_log_score_grad(w):
        combined = w @ exp_elpd  # (N,)
        combined = np.clip(combined, 1e-30, None)
        # grad_k = -sum_i exp_elpd_ki / combined_i
        grad = -np.sum(exp_elpd / combined[np.newaxis, :], axis=1)
        return grad

    # Initial weights: uniform
    w0 = np.ones(K) / K
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * K

    result = minimize(neg_log_score, w0, jac=neg_log_score_grad,
                      method='SLSQP', constraints=constraints, bounds=bounds,
                      options={'maxiter': 1000, 'ftol': 1e-10})

    weights = result.x
    weights = np.clip(weights, 0, None)
    weights = weights / weights.sum()
    return weights


def _pseudo_bma_weights(elpd_matrix: "np.ndarray") -> "np.ndarray":
    """Compute pseudo-BMA weights (Akaike-type weighting).

    w_k = exp(elpd_k) / sum_k exp(elpd_k)
    where elpd_k = sum_i elpd_k_i

    Args:
        elpd_matrix: (K, N) array of pointwise elpds.

    Returns:
        Array of shape (K,) with pseudo-BMA weights.
    """
    import numpy as np

    elpd_totals = elpd_matrix.sum(axis=1)  # (K,)
    elpd_max = elpd_totals.max()
    weights = np.exp(elpd_totals - elpd_max)
    weights = weights / weights.sum()
    return weights


# =============================================================================
# Log-likelihood extraction from any NumPyro model (post-fitting)
# =============================================================================

def compute_log_likelihood(model, posterior_samples, *model_args,
                           obs_name=None, **model_kwargs):
    """Compute pointwise log-likelihood by replaying a fitted model.

    This works with ANY NumPyro model — no need to modify the model
    definition. It traces the model with each posterior sample and
    extracts the log-probability of the observed site(s).

    Args:
        model: The NumPyro model function (same one passed to MCMC).
        posterior_samples: Dict of {name: array} with shape
            (num_chains, num_samples, ...) or (num_samples, ...).
            Typically from sampler.get_samples(group_by_chain=True).
        *model_args: Positional arguments passed to the model
            (same as during fitting).
        obs_name: Name(s) of the observed sample site(s). If None,
            auto-detects all sites that have obs= set.
            Can be a string or list of strings.
        **model_kwargs: Keyword arguments passed to the model
            (same as during fitting).

    Returns:
        If obs_name is a single string or auto-detected as one site:
            jnp.ndarray of shape (num_chains, num_samples, n_data_points)
        If multiple observed sites:
            dict of {site_name: jnp.ndarray} each shaped
            (num_chains, num_samples, n_data_points)

    Example:
        # After fitting:
        mcmc.run(rng_key, X, y)
        posterior = mcmc.get_samples(group_by_chain=True)

        # Compute log-likelihood (pass same args as model):
        ll = compute_log_likelihood(my_model, posterior, X, y)

        # Now use it:
        result = loo(ll)
        print(result)
    """
    import numpyro
    from numpyro import handlers
    import numpyro.distributions as dist

    # --- Flatten chains if present ---
    first_key = list(posterior_samples.keys())[0]
    first_val = posterior_samples[first_key]
    has_chains = False
    num_chains, num_samples = 1, first_val.shape[0]

    # Heuristic: if first dim is small (1-20) and second dim is large,
    # it's likely (chains, samples, ...). But we also check if all params
    # share the same first two dims.
    if first_val.ndim >= 2 and first_val.shape[0] <= 20:
        # Check consistency across all parameters
        shapes_consistent = all(
            v.ndim >= 2 and v.shape[0] == first_val.shape[0]
            for v in posterior_samples.values()
        )
        if shapes_consistent:
            has_chains = True
            num_chains = first_val.shape[0]
            num_samples = first_val.shape[1]

    # Flatten to (total_samples, ...) for tracing
    flat_samples = {}
    if has_chains:
        for k, v in posterior_samples.items():
            flat_samples[k] = v.reshape((-1,) + v.shape[2:])
    else:
        flat_samples = dict(posterior_samples)

    total_samples = flat_samples[first_key].shape[0]

    # --- Discover observed sites if obs_name not given ---
    if obs_name is None:
        # Trace the model once to find observed sites
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model).get_trace(*model_args, **model_kwargs)
        obs_sites = [
            name for name, site in trace.items()
            if site['type'] == 'sample'
            and site.get('is_observed', False)
        ]
        if not obs_sites:
            raise ValueError(
                "No observed sites found in model trace. "
                "Pass obs_name explicitly."
            )
    elif isinstance(obs_name, str):
        obs_sites = [obs_name]
    else:
        obs_sites = list(obs_name)

    # --- Compute log-likelihood for each posterior sample ---
    # We use numpyro's log_likelihood utility which is vectorized
    from numpyro.infer import log_likelihood as numpyro_log_likelihood

    log_liks = numpyro_log_likelihood(
        model, flat_samples, *model_args, **model_kwargs
    )
    # log_liks is dict of {site_name: (total_samples, n_data_points)}

    # Filter to requested sites
    log_liks = {k: v for k, v in log_liks.items() if k in obs_sites}

    # Reshape back to (chains, samples, n_data_points) if needed
    if has_chains:
        log_liks = {
            k: v.reshape(num_chains, num_samples, *v.shape[1:])
            for k, v in log_liks.items()
        }

    # Return single array if only one observed site
    if len(log_liks) == 1:
        return list(log_liks.values())[0]
    return log_liks


# =============================================================================
# iter_expanded — expand parameters to (label, 1D_samples) pairs
# Used by patch_diag.py for plotting and filtering
# =============================================================================

def iter_expanded(posterior_samples: dict, var_names=None, exclude_vars=None,
                  filter_regex=None):
    """Iterate over parameters, expanding multi-dimensional arrays into scalar entries.

    For a scalar param 'a' with shape (N,), yields ('a', flat_1D_array).
    For a vector param 'mu' with shape (N, K), yields ('mu[0]', ...), ('mu[K-1]', ...).

    Supports flat input (N, ...) from m.posteriors (group_by_chain=False).

    Args:
        posterior_samples: Dict of {name: array} with shape (N, ...) or (C, S, ...).
        var_names: If given, only include variables whose expanded label matches.
        exclude_vars: If given, exclude variables whose expanded label matches.
        filter_regex: If given, include only labels matching this regex.

    Yields:
        (label, 1D_samples) tuples where 1D_samples is a flat numpy array.
    """
    import numpy as np
    import re as _re

    if isinstance(var_names, str):
        var_names = [var_names]
    if isinstance(exclude_vars, str):
        exclude_vars = [exclude_vars]

    for name, samples in posterior_samples.items():
        samples = np.asarray(samples)

        # Flatten chain dim if chain-structured (C, S, ...) → (N, ...)
        if samples.ndim >= 3:
            C, S = samples.shape[0], samples.shape[1]
            samples = samples.reshape(C * S, *samples.shape[2:])

        # Determine parameter shape (dims beyond the sample dim)
        if samples.ndim == 1:
            param_shape = ()
        else:
            param_shape = samples.shape[1:]

        if not param_shape:
            entries = [(name, samples.flatten())]
        else:
            entries = []
            for idx in np.ndindex(param_shape):
                label = f"{name}[{', '.join(map(str, idx))}]"
                element = samples[(slice(None),) + idx].flatten()
                entries.append((label, element))

        for label, flat_samples in entries:
            # Filtering: match against full label or base name
            base = label.split('[')[0]
            if var_names is not None:
                if not any(label == v or base == v or label.startswith(v + '[')
                           for v in var_names):
                    continue
            if exclude_vars is not None:
                if any(label == v or base == v or label.startswith(v + '[')
                       for v in exclude_vars):
                    continue
            if filter_regex is not None and not _re.search(filter_regex, label):
                continue
            yield label, flat_samples


def compute_log_likelihood_manual(model, posterior_samples, *model_args,
                                  obs_name=None, batch_size=100,
                                  **model_kwargs):
    """Compute pointwise log-likelihood sample by sample.

    Slower fallback for models where numpyro.infer.log_likelihood
    doesn't work (e.g. complex control flow, custom distributions).
    Traces the model once per posterior sample.

    Args:
        model: The NumPyro model function.
        posterior_samples: Dict with shape (chains, samples, ...) or
            (samples, ...).
        *model_args: Args passed to model.
        obs_name: Observed site name(s). Auto-detected if None.
        batch_size: Process this many samples at a time (memory control).
        **model_kwargs: Kwargs passed to model.

    Returns:
        Same as compute_log_likelihood.
    """
    from numpyro import handlers

    # Flatten chains
    first_key = list(posterior_samples.keys())[0]
    first_val = posterior_samples[first_key]
    has_chains = False
    num_chains = 1

    if first_val.ndim >= 2 and first_val.shape[0] <= 20:
        shapes_consistent = all(
            v.ndim >= 2 and v.shape[0] == first_val.shape[0]
            for v in posterior_samples.values()
        )
        if shapes_consistent:
            has_chains = True
            num_chains = first_val.shape[0]

    flat_samples = {}
    if has_chains:
        for k, v in posterior_samples.items():
            flat_samples[k] = v.reshape((-1,) + v.shape[2:])
    else:
        flat_samples = dict(posterior_samples)

    total_samples = flat_samples[first_key].shape[0]

    # Discover obs sites
    if obs_name is None:
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model).get_trace(*model_args, **model_kwargs)
        obs_sites = [
            name for name, site in trace.items()
            if site['type'] == 'sample' and site.get('is_observed', False)
        ]
        if not obs_sites:
            raise ValueError("No observed sites found. Pass obs_name explicitly.")
    elif isinstance(obs_name, str):
        obs_sites = [obs_name]
    else:
        obs_sites = list(obs_name)

    # Trace sample by sample
    log_liks = {site: [] for site in obs_sites}

    for s in range(total_samples):
        # Get single sample
        single_sample = {k: v[s] for k, v in flat_samples.items()}

        # Condition model on this posterior sample and trace
        conditioned = handlers.condition(model, data=single_sample)
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(conditioned).get_trace(
                *model_args, **model_kwargs
            )

        for site in obs_sites:
            site_data = trace[site]
            fn = site_data['fn']
            value = site_data['value']
            ll_i = fn.log_prob(value)
            log_liks[site].append(ll_i)

    # Stack
    log_liks = {k: jnp.stack(v) for k, v in log_liks.items()}

    # Reshape back to (chains, samples, ...)
    if has_chains:
        num_samples = total_samples // num_chains
        log_liks = {
            k: v.reshape(num_chains, num_samples, *v.shape[1:])
            for k, v in log_liks.items()
        }

    if len(log_liks) == 1:
        return list(log_liks.values())[0]
    return log_liks
