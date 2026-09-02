"""Pure-numpy statistics for the Workflow class's recovery and SBC loops.

Kept free of BF/JAX imports on purpose: these functions only ever see plain
Python/numpy values pulled out of posterior draws by ``core.py``, so they can
run inside a worker process (multiprocessing) without pulling in a second
copy of JAX/XLA there.
"""
import numpy as np


def recovery_metrics(table, param_names, coverage_target=(0.85, 0.93),
                      coverage_soft=(0.80, 0.95)):
    """Per-parameter bias / RMSE / R^2 / coverage / Weighted Recovery Grade.

    Mirrors the grading scheme used throughout the BF workflow guides
    (``model_improvement`` / ``dgp_estimation``):
        WRG = 0.6 * R^2 + 0.4 * coverage_score
        Grade A (>=0.85), B (>=0.75), F (<0.75)

    Args:
        table: DataFrame with ``{name}_true``, ``{name}_mean``,
            ``{name}_covered`` columns per parameter (see ``core.recover``).
        param_names: Parameters to score.
        coverage_target: (lo, hi) fraction range scoring 1.0 (default 0.85-0.93
            for an 89% HDI).
        coverage_soft: (lo, hi) wider range scoring 0.5; outside it scores 0.0.

    Returns:
        dict[name -> dict(bias, rmse, r2, coverage, wrg, grade)].
    """
    lo_t, hi_t = coverage_target
    lo_s, hi_s = coverage_soft
    metrics = {}
    for name in param_names:
        true = np.asarray(table[f"{name}_true"], dtype=float)
        est = np.asarray(table[f"{name}_mean"], dtype=float)
        covered = np.asarray(table[f"{name}_covered"], dtype=bool)

        diff = est - true
        bias = float(diff.mean()) if len(diff) else float("nan")
        rmse = float(np.sqrt((diff ** 2).mean())) if len(diff) else float("nan")

        ss_res = float(((true - est) ** 2).sum())
        ss_tot = float(((true - true.mean()) ** 2).sum()) if len(true) else 0.0
        r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        coverage = float(covered.mean()) if len(covered) else float("nan")
        if lo_t <= coverage <= hi_t:
            coverage_score = 1.0
        elif lo_s <= coverage < lo_t or hi_t < coverage <= hi_s:
            coverage_score = 0.5
        else:
            coverage_score = 0.0

        r2_clamped = max(r2, 0.0) if not np.isnan(r2) else 0.0
        wrg = 0.6 * r2_clamped + 0.4 * coverage_score
        grade = "A" if wrg >= 0.85 else ("B" if wrg >= 0.75 else "F")

        metrics[name] = dict(bias=bias, rmse=rmse, r2=r2, coverage=coverage,
                              wrg=wrg, grade=grade)
    return metrics


def sbc_uniformity(table, param_names, n_post_draws, n_bins=None):
    """Chi-square test of rank-uniformity for each parameter's SBC ranks.

    A well-calibrated model/sampler should produce ranks that are (discrete-)
    uniform on [0, n_post_draws]. This bins the observed ranks and compares
    to the uniform expectation via a chi-square goodness-of-fit statistic.

    Args:
        table: DataFrame with a ``{name}_rank`` column per parameter.
        param_names: Parameters to test.
        n_post_draws: Number of posterior draws each rank was computed against
            (the ranks live in [0, n_post_draws]).
        n_bins: Histogram bins for the test. Defaults to
            ``min(20, max(5, n_replications // 5))``.

    Returns:
        dict[name -> dict(statistic, p_value, n_bins)]. p_value <= 0.05 is
        evidence the ranks are NOT uniform -- i.e. a calibration failure.
    """
    from scipy import stats as _stats

    results = {}
    for name in param_names:
        ranks = np.asarray(table[f"{name}_rank"], dtype=float)
        n = len(ranks)
        bins = n_bins or min(20, max(5, n // 5)) if n else 1
        bins = max(bins, 1)
        hist, _ = np.histogram(ranks, bins=bins, range=(0, n_post_draws))
        expected = n / bins if bins else float("nan")
        if expected > 0:
            chi2 = float(((hist - expected) ** 2 / expected).sum())
            p_value = float(1.0 - _stats.chi2.cdf(chi2, df=max(bins - 1, 1)))
        else:
            chi2, p_value = float("nan"), float("nan")
        results[name] = dict(statistic=chi2, p_value=p_value, n_bins=bins)
    return results
