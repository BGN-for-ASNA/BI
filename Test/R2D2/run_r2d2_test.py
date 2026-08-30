#!/usr/bin/env python3
# ======================================================================
# R2D2 prior — implementation-correctness test  (BayesForge / BI)
# ----------------------------------------------------------------------
# Ref doc : BF/Documentation/37.Feature extraction: R2D2.qmd
# Prior   : R^2 ~ Beta(a, b)            (Zhang et al. 2022, "R2D2")
#           tau^2 = sigma^2 * R^2 / (1 - R^2)      (global explained var)
#           phi   ~ Dirichlet(1, ..., 1)          (V-simplex weights)
#           lam_j^2 = tau^2 * phi_j               (local variances)
#           beta_j ~ Normal(0, lam_j^2)
#           y      ~ Normal(alpha + X beta, sigma)
#
# What this script does (Bayesian-inference workflow):
#   1. Simulate N individuals, V standardized Normal covariates, of which
#      only `n_strong` carry a real effect  -> generate y.
#   2. Compute the *model* R^2  =  Var(X beta) / (Var(X beta) + sigma^2)
#      (the quantity the Beta prior is placed on, see qmd "Definition of R^2"),
#      plus the classical sample R^2 = 1 - SSres/SStot for reference.
#   3. Prior-predictive check on R^2 for every Beta(a,b) setting.
#   4. Fit the R2D2 model with NUTS for every Beta(a,b) setting.
#   5. Compare posterior estimates against the simulation truth:
#        - recovery of the `n_strong` real coefficients (bias, RMSE, HDI cover)
#        - shrinkage of the null coefficients (mean |beta|, HDI false positives)
#        - recovery of R^2, sigma
#        - MCMC health (max R-hat, min ESS, divergences)
#   6. Vary the variance prior  R^2 ~ Beta(1/3, 3), Beta(1,1), ...  and
#      report whether/how the estimates move (prior-sensitivity, the
#      "When R2D2 Fails" section of the qmd).
#
# All console output is also written to logs/r2d2_test_<stamp>.log ;
# machine-readable results -> out/results_<stamp>.json ;
# figures -> out/*.png
# ======================================================================
from __future__ import annotations

import os
import sys

# Force CPU: the local box has a GPU jax whose CuDNN is mismatched (9.1 runtime
# vs 9.8 build). This test is small and CPU-only by design. Must be set before
# jax / BayesForge import.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Give XLA enough CPU "devices" for parallel MCMC chains. Must be set before jax
# initializes. Pre-scan argv for --num-chains so the mesh matches the request.
def _prescan_chains(default=4):
    a = sys.argv
    for i, t in enumerate(a):
        if t == "--num-chains" and i + 1 < len(a):
            return max(1, int(a[i + 1]))
        if t.startswith("--num-chains="):
            return max(1, int(t.split("=", 1)[1]))
    return default

_NCHAINS = _prescan_chains()
os.environ.setdefault(
    "XLA_FLAGS",
    f"--xla_force_host_platform_device_count={_NCHAINS}",
)

import argparse
import hashlib
import json
import logging
import platform as _platform
import socket
import sys
import time
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
OUT_DIR = os.path.join(HERE, "out")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
log = logging.getLogger("r2d2")


# ----------------------------------------------------------------------
# logging
# ----------------------------------------------------------------------
def setup_logging(stamp: str) -> str:
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    path = os.path.join(LOG_DIR, f"r2d2_test_{stamp}.log")
    fh = logging.FileHandler(path, mode="w")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return path


# ----------------------------------------------------------------------
# 1. simulation
# ----------------------------------------------------------------------
def simulate(N, V, n_strong, sigma_true, beta_scale, seed):
    """N individuals, V standardized N(0,1) covariates, `n_strong` real effects."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(N, V))
    # scale (standardize) each covariate  -> R2D2 requires standardized predictors
    X = (X - X.mean(0)) / X.std(0)

    beta_true = np.zeros(V)
    strong_idx = np.sort(rng.choice(V, size=n_strong, replace=False))
    signs = rng.choice([-1.0, 1.0], size=n_strong)
    mags = beta_scale * (1.0 + 0.4 * rng.standard_normal(n_strong))  # ~ beta_scale
    beta_true[strong_idx] = signs * np.abs(mags)

    alpha_true = 0.0
    signal = X @ beta_true
    y = alpha_true + signal + rng.normal(0.0, sigma_true, size=N)

    # --- R^2 definitions -------------------------------------------------
    var_signal = float(np.var(signal))
    # model / explained-variance R^2  (the quantity Beta(a,b) is a prior on;
    #   equivalently  tau^2 / (tau^2 + sigma^2)  with  tau^2 = Var(X beta) )
    r2_model = var_signal / (var_signal + sigma_true ** 2)
    # classical sample R^2 for this particular draw
    ss_res = float(np.sum((y - (alpha_true + signal)) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_sample = 1.0 - ss_res / ss_tot

    return dict(
        X=X.astype("float64"), y=y.astype("float64"),
        beta_true=beta_true, alpha_true=alpha_true, sigma_true=float(sigma_true),
        strong_idx=strong_idx.astype(int),
        r2_model=float(r2_model), r2_sample=float(r2_sample),
        var_signal=var_signal,
    )


# ----------------------------------------------------------------------
# 2. R2D2 model factory   (mirrors the qmd example, vectorized over V)
# ----------------------------------------------------------------------
def make_r2d2_model(m, a, b, centered=False, dir_conc=1.0):
    """R2D2 prior exactly as in 37.Feature extraction: R2D2.qmd.

    centered=True  reproduces the qmd snippet verbatim: beta_j ~ Normal(0, lam_j).
    centered=False (default) uses the mathematically identical non-centered form
    beta_j = z_j * lam_j, z_j ~ Normal(0,1) -- needed here because V > N drives
    the centered parameterization into a funnel (severe divergences / R-hat) for
    priors that put mass on R^2 -> 1.
    """
    import jax.numpy as jnp
    import numpyro

    def model(X, y):
        V = X.shape[1]
        sigma = m.dist.exponential(1.0, name="sigma")          # sigma ~ Exp(1)
        r2 = m.dist.beta(a, b, name="r2")                       # R^2 ~ Beta(a,b)
        r2c = jnp.clip(r2, 1e-4, 1.0 - 1e-4)                    # guard 1/(1-r2)
        tau2 = sigma ** 2 * r2c / (1.0 - r2c)                   # global explained var
        phi = m.dist.dirichlet(jnp.full(V, dir_conc), name="phi")  # simplex weights; conc<1 -> sparse
        lam2 = tau2 * phi                                       # local variances
        alpha = m.dist.normal(0.0, 2.0, name="alpha")
        if centered:
            beta = m.dist.normal(jnp.zeros(V), jnp.sqrt(lam2), name="beta")     # (V,)
        else:
            z = m.dist.normal(jnp.zeros(V), 1.0, name="beta_z")                 # (V,)
            beta = numpyro.deterministic("beta", z * jnp.sqrt(lam2))
        mu = alpha + X @ beta
        m.dist.normal(mu, sigma, obs=y, name="y_obs")

    return model


# ----------------------------------------------------------------------
# 3. prior-predictive check on R^2  (Beta(a,b) is analytic)
# ----------------------------------------------------------------------
def prior_predictive_r2(a, b, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    r2 = rng.beta(a, b, size=n)
    mean = a / (a + b)
    return dict(a=a, b=b, prior_mean=float(mean),
                q05=float(np.quantile(r2, 0.05)),
                q50=float(np.quantile(r2, 0.50)),
                q95=float(np.quantile(r2, 0.95)),
                p_gt_0_5=float(np.mean(r2 > 0.5)))


# ----------------------------------------------------------------------
# helpers for posterior summaries
# ----------------------------------------------------------------------
def diag_summary(posteriors_by_chain):
    """max R-hat / min ESS over all scalar+vector sites, via numpyro."""
    from numpyro.diagnostics import summary as nsummary
    s = nsummary(posteriors_by_chain, prob=0.89, group_by_chain=True)
    max_rhat, min_ess = 0.0, np.inf
    for _, d in s.items():
        max_rhat = max(max_rhat, float(np.nanmax(d["r_hat"])))
        min_ess = min(min_ess, float(np.nanmin(d["n_eff"])))
    return float(max_rhat), float(min_ess)


def hdi(samples, prob=0.89):
    from numpyro.diagnostics import hpdi
    lo, hi = hpdi(samples, prob=prob, axis=0)
    return np.asarray(lo), np.asarray(hi)


# ----------------------------------------------------------------------
# 4-5. fit one Beta(a,b) configuration and score it against truth
# ----------------------------------------------------------------------
def fit_config(m, sim, a, b, args):
    import jax.numpy as jnp

    label = f"Beta({_fmt(a)},{_fmt(b)})"
    log.info("-" * 70)
    log.info("FIT  R^2 ~ %s   (prior mean R^2 = %.3f)", label, a / (a + b))

    m.data_on_model = {"X": jnp.asarray(sim["X"]), "y": jnp.asarray(sim["y"])}
    model = make_r2d2_model(m, a, b, centered=args.centered, dir_conc=args.dir_conc)

    fit_seed = _cfg_seed(args.seed, sim["sigma_true"], a, b)
    t0 = time.time()
    m.fit(model,
          num_warmup=args.num_warmup,
          num_samples=args.num_samples,
          num_chains=args.num_chains,
          target_accept_prob=args.target_accept,
          max_tree_depth=args.max_tree_depth,
          progress_bar=False,
          seed=fit_seed,
          extra_fields=("diverging",))
    dt = time.time() - t0

    post = m.posteriors                       # flat: name -> (S, ...)
    beta_s = np.asarray(post["beta"])         # (S, V)
    r2_s = np.asarray(post["r2"]).ravel()
    sig_s = np.asarray(post["sigma"]).ravel()

    beta_mean = beta_s.mean(0)
    b_lo, b_hi = hdi(beta_s, prob=0.89)

    strong = sim["strong_idx"]
    null = np.setdiff1d(np.arange(sim["X"].shape[1]), strong)
    bt = sim["beta_true"]

    # recovery of the real effects
    err = beta_mean[strong] - bt[strong]
    rmse_strong = float(np.sqrt(np.mean(err ** 2)))
    bias_strong = float(np.mean(err))
    # signed shrinkage toward 0 (|est| < |true| ?)
    shrink = float(np.mean(np.abs(beta_mean[strong]) / np.abs(bt[strong])))
    cover_strong = (bt[strong] >= b_lo[strong]) & (bt[strong] <= b_hi[strong])
    cover_rate = float(np.mean(cover_strong))

    # shrinkage of the nulls
    null_mean_abs = float(np.mean(np.abs(beta_mean[null])))
    null_max_abs = float(np.max(np.abs(beta_mean[null])))
    null_fp = int(np.sum((b_lo[null] > 0) | (b_hi[null] < 0)))   # 89% HDI excludes 0

    # R^2 / sigma recovery
    r2_pm = float(r2_s.mean())
    r2_lo, r2_hi = hdi(r2_s, prob=0.89)
    sig_pm = float(sig_s.mean())

    max_rhat, min_ess = diag_summary(m.posteriors_by_chain)
    try:
        div = int(np.sum(np.asarray(m.sampler.get_extra_fields()["diverging"])))
    except Exception as e:                    # pragma: no cover
        log.warning("could not read divergences: %s", e)
        div = -1

    # BI-workflow style summary table (also logs the native BF summary)
    try:
        bf_tab = m.summary(include=["r2", "sigma", "alpha"])
        log.info("BF summary (r2, sigma, alpha):\n%s", bf_tab)
    except Exception as e:
        log.warning("m.summary() failed: %s", e)

    log.info("R^2  true(model)=%.3f  true(sample)=%.3f  ->  post mean=%.3f  "
             "HDI89=[%.3f, %.3f]", sim["r2_model"], sim["r2_sample"],
             r2_pm, float(r2_lo), float(r2_hi))
    log.info("sigma true=%.3f  ->  post mean=%.3f", sim["sigma_true"], sig_pm)
    log.info("strong beta : RMSE=%.3f  bias=%.3f  mean|est|/|true|=%.2f  "
             "HDI89 coverage=%.0f%% (%d/%d)", rmse_strong, bias_strong, shrink,
             100 * cover_rate, int(cover_strong.sum()), len(strong))
    log.info("null   beta : mean|est|=%.4f  max|est|=%.3f  HDI89 false-pos=%d/%d",
             null_mean_abs, null_max_abs, null_fp, len(null))
    log.info("MCMC : max R-hat=%.3f  min ESS=%.0f  divergences=%d  (%.1fs)",
             max_rhat, min_ess, div, dt)

    per_strong = [
        dict(idx=int(j), beta_true=float(bt[j]), post_mean=float(beta_mean[j]),
             hdi_lo=float(b_lo[j]), hdi_hi=float(b_hi[j]),
             covered=bool(cover_strong[k]))
        for k, j in enumerate(strong)
    ]

    return dict(
        label=label, a=float(a), b=float(b), prior_mean_r2=float(a / (a + b)),
        mcmc_seed=int(fit_seed), runtime_s=dt,
        r2_true_model=sim["r2_model"], r2_true_sample=sim["r2_sample"],
        r2_post_mean=r2_pm, r2_hdi=[float(r2_lo), float(r2_hi)],
        sigma_true=sim["sigma_true"], sigma_post_mean=sig_pm,
        rmse_strong=rmse_strong, bias_strong=bias_strong,
        shrink_ratio_strong=shrink, coverage_strong=cover_rate,
        null_mean_abs=null_mean_abs, null_max_abs=null_max_abs,
        null_false_pos=null_fp, n_null=len(null),
        max_rhat=max_rhat, min_ess=min_ess, divergences=div,
        per_strong=per_strong,
        _beta_mean=beta_mean, _b_lo=b_lo, _b_hi=b_hi, _r2_s=r2_s,   # for plots
    )


def _fmt(x):
    return f"{x:.3g}"


def _cfg_seed(base: int, sigma_true: float, a: float, b: float) -> int:
    """Deterministic per-fit MCMC seed.

    The NUTS run must be reproducible from ``--seed`` (the DGP already is), yet
    each (noise level, Beta prior) fit needs its own independent stream — reusing
    one key across all fits would couple their divergence counts. Derive a stable
    32-bit seed from the base seed and the config tuple.
    """
    key = f"{base}|{sigma_true:.6g}|{a:.6g}|{b:.6g}"
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    return (base + h) & 0x7FFFFFFF


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------
def make_figures(sim, results, stamp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bt = sim["beta_true"]
    strong = sim["strong_idx"]
    null = np.setdiff1d(np.arange(len(bt)), strong)
    nC = len(results)

    # ---- Fig 1: true vs posterior-mean coefficients, per prior ----------
    fig, axes = plt.subplots(1, nC, figsize=(4.2 * nC, 4.0), squeeze=False)
    for ax, r in zip(axes[0], results):
        bm, lo, hi = r["_beta_mean"], r["_b_lo"], r["_b_hi"]
        ax.errorbar(bt[strong], bm[strong],
                    yerr=[bm[strong] - lo[strong], hi[strong] - bm[strong]],
                    fmt="o", ms=6, capsize=3, color="C3", label="strong (true!=0)")
        ax.scatter(bt[null], bm[null], s=10, alpha=.4, color="C0", label="null (true=0)")
        lim = max(3.0, np.abs(bt).max() * 1.15)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(f"{r['label']}\nprior E[R2]={r['prior_mean_r2']:.2f} | "
                     f"shrink={r['shrink_ratio_strong']:.2f}")
        ax.set_xlabel("true beta"); ax.set_ylabel("posterior mean beta")
        ax.legend(fontsize=8)
    fig.suptitle(f"R2D2 coefficient recovery  (N={sim['X'].shape[0]}, "
                 f"V={sim['X'].shape[1]}, true model R2={sim['r2_model']:.2f})")
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, f"beta_recovery_{stamp}.png")
    fig.savefig(p1, dpi=110); plt.close(fig)

    # ---- Fig 2: posterior of R^2 per prior -----------------------------
    fig, ax = plt.subplots(figsize=(1.7 * nC + 2, 4.2))
    data = [r["_r2_s"] for r in results]
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    ax.axhline(sim["r2_model"], color="C3", lw=2, label=f"true model R2={sim['r2_model']:.3f}")
    ax.axhline(sim["r2_sample"], color="C2", ls=":", lw=1.5,
               label=f"true sample R2={sim['r2_sample']:.3f}")
    ax.set_xticks(range(1, nC + 1))
    ax.set_xticklabels([r["label"] for r in results], rotation=20, ha="right")
    ax.set_ylabel("posterior R^2"); ax.set_ylim(0, 1)
    ax.set_title("R2D2: posterior of R^2 vs simulation truth, by variance prior")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, f"r2_posterior_{stamp}.png")
    fig.savefig(p2, dpi=110); plt.close(fig)

    # ---- Fig 3: error metrics per prior ------------------------------
    fig, ax = plt.subplots(figsize=(1.7 * nC + 2, 4.2))
    x = np.arange(nC)
    ax.bar(x - 0.2, [r["rmse_strong"] for r in results], 0.4, label="RMSE strong beta")
    ax.bar(x + 0.2, [r["null_mean_abs"] for r in results], 0.4, label="mean |beta| null")
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in results], rotation=20, ha="right")
    ax.set_ylabel("value"); ax.set_title("R2D2 prior sensitivity: recovery error")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p3 = os.path.join(OUT_DIR, f"error_metrics_{stamp}.png")
    fig.savefig(p3, dpi=110); plt.close(fig)

    return [p1, p2, p3]


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def parse_priors(s):
    out = []
    for tok in s.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        a, b = tok.split(",")
        out.append((float(eval(a)), float(eval(b))))   # allow "1/3"
    return out


def parse_grid(s, fallback):
    s = (s or "").strip()
    if not s:
        return [float(fallback)]
    return [float(eval(t)) for t in s.replace(";", ",").split(",") if t.strip()]


def run_scenario(m, priors, sigma_true, args, tag):
    """One DGP noise level: simulate -> fit every Beta(a,b) -> score vs truth."""
    sim = simulate(args.N, args.V, args.n_strong, sigma_true, args.beta_scale, args.seed)
    log.info("=" * 70)
    log.info("SCENARIO %s :  DGP noise sigma_true = %.3f", tag, sigma_true)
    log.info("Var(X beta)=%.3f   sigma_true^2=%.3f", sim["var_signal"], sigma_true ** 2)
    log.info("model  R^2 = Var(Xb)/(Var(Xb)+sigma^2) = %.4f", sim["r2_model"])
    log.info("sample R^2 = 1 - SSres/SStot          = %.4f", sim["r2_sample"])
    log.info("true strong betas = %s",
             np.round(sim["beta_true"][sim["strong_idx"]], 3).tolist())

    results = []
    for a, b in priors:
        try:
            results.append(fit_config(m, sim, a, b, args))
        except Exception as e:
            log.exception("fit failed for Beta(%s,%s) at sigma=%.2f: %s", a, b, sigma_true, e)

    log.info("-" * 70)
    log.info("PRIOR SENSITIVITY @ sigma_true=%.3f (true model R^2=%.3f)",
             sigma_true, sim["r2_model"])
    hdr = (f"{'prior':>14} | {'E[R2]pri':>8} | {'R2 post':>8} | {'sigma':>6} | "
           f"{'RMSE_s':>7} | {'shrink':>6} | {'cov_s':>6} | {'|b|null':>8} | "
           f"{'FP':>3} | {'Rhat':>5} | {'ESS':>6} | {'div':>4}")
    log.info(hdr)
    log.info("-" * len(hdr))
    for r in results:
        log.info(f"{r['label']:>14} | {r['prior_mean_r2']:8.3f} | "
                 f"{r['r2_post_mean']:8.3f} | {r['sigma_post_mean']:6.3f} | "
                 f"{r['rmse_strong']:7.3f} | {r['shrink_ratio_strong']:6.2f} | "
                 f"{r['coverage_strong']*100:5.0f}% | {r['null_mean_abs']:8.4f} | "
                 f"{r['null_false_pos']:3d} | {r['max_rhat']:5.2f} | "
                 f"{r['min_ess']:6.0f} | {r['divergences']:4d}")

    # A clean NUTS fit has ~0 divergences. Flag anything above ~0.5% of the
    # post-warmup transitions: that is the R2D2 funnel pathology this test
    # exists to surface ("When R2D2 Fails" in the qmd), and R-hat alone stays
    # near 1.0 while it happens. The old 10% gate passed fits with hundreds of
    # divergent transitions.
    total_draws = args.num_samples * args.num_chains
    thr = max(1, int(round(0.005 * total_draws)))
    ok = [r for r in results
          if r["max_rhat"] <= 1.1 and 0 <= r["divergences"] <= thr]
    ok_ids = {id(r) for r in ok}                     # dicts hold ndarrays: no `in`
    bad = [r for r in results if id(r) not in ok_ids]
    if bad:
        log.warning("NON-CONVERGED @ sigma=%.2f (divergence gate = %d / %d transitions): %s",
                    sigma_true, thr, total_draws,
                    ", ".join(f"{r['label']}(Rhat={r['max_rhat']:.2f},div={r['divergences']})"
                              for r in bad))
    if ok:
        spread = max(r["r2_post_mean"] for r in ok) - min(r["r2_post_mean"] for r in ok)
        sspread = max(r["sigma_post_mean"] for r in ok) - min(r["sigma_post_mean"] for r in ok)
        log.info("across converged priors: R^2 post spread=%.3f  sigma post spread=%.3f  -> %s",
                 spread, sspread, "PRIOR-SENSITIVE" if spread > 0.05 else "robust")

    try:
        for p in make_figures(sim, results, f"{STAMP}_{tag}"):
            log.info("figure -> %s", p)
    except Exception as e:
        log.exception("scenario figure failed: %s", e)

    return sim, results


def make_sweep_figure(scenarios, stamp):
    """Cross-scenario: recovery metrics vs the DGP noise level / true R^2."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sig = [s["sim"]["sigma_true"] for s in scenarios]
    r2t = [s["sim"]["r2_model"] for s in scenarios]
    rmaps = [{r["label"]: r for r in s["results"]} for s in scenarios]
    # union of labels seen across scenarios, first-seen order
    labels = list(dict.fromkeys(l for rm in rmaps for l in rm))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for lab in labels:
        # scenarios where this prior actually produced a fit
        idx = [i for i, rm in enumerate(rmaps) if lab in rm]
        if not idx:
            continue
        rmse = [rmaps[i][lab]["rmse_strong"] for i in idx]
        r2e = [rmaps[i][lab]["r2_post_mean"] - r2t[i] for i in idx]
        sige = [rmaps[i][lab]["sigma_post_mean"] / sig[i] - 1.0 for i in idx]
        axes[0].plot([sig[i] for i in idx], rmse, "o-", label=lab)
        axes[1].plot([r2t[i] for i in idx], r2e, "o-", label=lab)
        axes[2].plot([r2t[i] for i in idx], sige, "o-", label=lab)
    axes[0].set_xlabel("DGP noise sigma_true"); axes[0].set_ylabel("RMSE strong beta")
    axes[0].set_title("coefficient recovery error vs noise")
    axes[1].axhline(0, color="k", lw=.8)
    axes[1].set_xlabel("true model R^2"); axes[1].set_ylabel("R^2 post mean - true")
    axes[1].set_title("R^2 recovery bias vs true R^2")
    axes[2].axhline(0, color="k", lw=.8)
    axes[2].set_xlabel("true model R^2"); axes[2].set_ylabel("sigma post mean / true - 1")
    axes[2].set_title("sigma recovery bias vs true R^2")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle(f"R2D2 recovery across DGP noise levels "
                 f"(N={scenarios[0]['sim']['X'].shape[0]}, V={scenarios[0]['sim']['X'].shape[1]})")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, f"noise_sweep_{stamp}.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description="R2D2 prior correctness test (BayesForge)")
    ap.add_argument("--N", type=int, default=50)
    ap.add_argument("--V", type=int, default=100)
    ap.add_argument("--n-strong", type=int, default=5)
    ap.add_argument("--sigma-true", type=float, default=1.0,
                    help="DGP residual noise sd (used when --sigma-grid is empty)")
    ap.add_argument("--sigma-grid", type=str, default="1,2,4,8",
                    help='comma list of DGP noise sd values to sweep; '
                         'empty -> just --sigma-true')
    ap.add_argument("--beta-scale", type=float, default=2.5,
                    help="approx magnitude of the strong coefficients")
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--priors", type=str,
                    default="1/3,3 ; 1,1 ; 3,3 ; 3,1 ; 5,1",
                    help='";"-separated Beta a,b pairs for R^2')
    ap.add_argument("--num-warmup", type=int, default=1000)
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--num-chains", type=int, default=4)
    ap.add_argument("--target-accept", type=float, default=0.9)
    ap.add_argument("--max-tree-depth", type=int, default=12)
    ap.add_argument("--dir-conc", type=float, default=1.0,
                    help="Dirichlet concentration a0 (phi ~ Dir(a0,...,a0)); "
                         "a0<1 favours sparse variance allocation (heavier-tailed "
                         "coefficient priors, more horseshoe-like)")
    ap.add_argument("--centered", action="store_true",
                    help="use the verbatim qmd parameterization beta~N(0,lam) "
                         "(default: non-centered, sampler-robust)")
    args = ap.parse_args()

    log_path = setup_logging(STAMP)
    log.info("=" * 70)
    log.info("R2D2 PRIOR CORRECTNESS TEST")
    log.info("host=%s  user=%s  python=%s", socket.gethostname(),
             os.environ.get("USER", "?"), _platform.python_version())
    log.info("cwd=%s", HERE)
    log.info("log file -> %s", log_path)

    # environment / BayesForge -----------------------------------------
    os.environ.setdefault("BF_QUIET", "0")
    try:
        from BayesForge import bf
        import BayesForge
        log.info("BayesForge %s", BayesForge.__version__)
    except Exception as e:
        log.error("cannot import BayesForge: %s", e)
        raise
    import jax
    log.info("jax %s  devices=%s", jax.__version__, jax.devices())

    priors = parse_priors(args.priors)
    sigma_grid = parse_grid(args.sigma_grid, args.sigma_true)
    log.info("config: N=%d V=%d n_strong=%d beta_scale=%.2f seed=%d",
             args.N, args.V, args.n_strong, args.beta_scale, args.seed)
    log.info("DGP noise sweep: sigma_true in %s", [round(s, 3) for s in sigma_grid])
    log.info("variance priors on R^2: %s",
             ", ".join(f"Beta({_fmt(a)},{_fmt(b)})" for a, b in priors))
    log.info("NUTS: warmup=%d samples=%d chains=%d target_accept=%.2f max_tree=%d",
             args.num_warmup, args.num_samples, args.num_chains,
             args.target_accept, args.max_tree_depth)

    # prior-predictive on R^2 (depends only on Beta(a,b), not on the DGP noise)
    log.info("=" * 70)
    log.info("PRIOR-PREDICTIVE CHECK ON R^2")
    ppc = []
    for a, b in priors:
        d = prior_predictive_r2(a, b, seed=args.seed)
        ppc.append(d)
        log.info("Beta(%-7s,%-4s): mean=%.3f  q05=%.3f q50=%.3f q95=%.3f  P(R2>0.5)=%.2f",
                 _fmt(a), _fmt(b), d["prior_mean"], d["q05"], d["q50"],
                 d["q95"], d["p_gt_0_5"])

    m = bf(platform="cpu", cores=args.num_chains, rand_seed=args.seed,
           print_devices_found=False)
    log.info("=" * 70)
    log.info("INFERENCE  (%d noise levels x %d priors = %d fits)",
             len(sigma_grid), len(priors), len(sigma_grid) * len(priors))

    scenarios = []
    for i, sg in enumerate(sigma_grid):
        tag = f"s{_fmt(sg).replace('.', 'p')}"
        sim, results = run_scenario(m, priors, sg, args, tag)
        scenarios.append(dict(sim=sim, results=results, tag=tag))

    # cross-noise summary --------------------------------------------
    # Wrapped: a formatting slip here must never cost us the results JSON below.
    # `labels` comes from the requested priors, not from scenarios[0] — a failed
    # fit leaves `results` shorter than the prior list, and every lookup is a
    # dict .get() so a missing (label, scenario) cell degrades to "n/a".
    try:
        log.info("=" * 70)
        log.info("NOISE SWEEP SUMMARY  (RMSE strong beta  /  sigma_post per sigma_true)")
        labels = [f"Beta({_fmt(a)},{_fmt(b)})" for a, b in priors]
        rmaps = [{r["label"]: r for r in s["results"]} for s in scenarios]
        hdr = f"{'sigma_true':>10} | {'R2_true':>8} | " + " | ".join(f"{l:>13}" for l in labels)
        log.info(hdr)
        log.info("-" * len(hdr))
        for s, rmap in zip(scenarios, rmaps):
            cells = []
            for l in labels:
                r = rmap.get(l)
                if r is None:
                    cells.append("n/a")
                    continue
                flag = "" if (r["max_rhat"] <= 1.1) else "!"
                cells.append(f"{r['rmse_strong']:.2f}/{r['sigma_post_mean']:.2f}{flag:>1}")
            log.info(f"{s['sim']['sigma_true']:10.3f} | {s['sim']['r2_model']:8.3f} | "
                     + " | ".join(f"{c:>13}" for c in cells))
        log.info("cell = RMSE(strong beta) / posterior-mean sigma   "
                 "('!' = max R-hat > 1.1 ; 'n/a' = fit failed)")

        log.info("-" * 70)
        log.info("does the DGP noise change the R2D2 conclusions?")
        for l in labels:
            got = [(rmap.get(l), s) for rmap, s in zip(rmaps, scenarios)]
            got = [(r, s) for r, s in got if r is not None]
            if len(got) < 2:
                log.info("%-14s : too few successful fits (%d) to compare", l, len(got))
                continue
            rmses = [r["rmse_strong"] for r, _ in got]
            sbias = [r["sigma_post_mean"] / s["sim"]["sigma_true"] - 1 for r, s in got]
            log.info("%-14s : RMSE %.2f -> %.2f as noise grows ; sigma bias %+.0f%% -> %+.0f%%",
                     l, rmses[0], rmses[-1], 100 * sbias[0], 100 * sbias[-1])
    except Exception as e:
        log.exception("noise-sweep summary failed (results JSON still written): %s", e)

    # figures + json ------------------------------------------------
    sweep_fig = None
    try:
        sweep_fig = make_sweep_figure(scenarios, STAMP)
        log.info("figure -> %s", sweep_fig)
    except Exception as e:
        log.exception("sweep figure failed: %s", e)

    res_path = os.path.join(OUT_DIR, f"results_{STAMP}.json")
    with open(res_path, "w") as f:
        json.dump(dict(
            stamp=STAMP,
            host=socket.gethostname(),
            args=vars(args),
            prior_predictive=ppc,
            scenarios=[dict(
                sigma_true=s["sim"]["sigma_true"],
                simulation=dict(
                    N=args.N, V=args.V, n_strong=args.n_strong,
                    strong_idx=s["sim"]["strong_idx"].tolist(),
                    beta_true_strong=s["sim"]["beta_true"][s["sim"]["strong_idx"]].tolist(),
                    var_signal=s["sim"]["var_signal"],
                    sigma_true=s["sim"]["sigma_true"],
                    r2_model=s["sim"]["r2_model"], r2_sample=s["sim"]["r2_sample"],
                ),
                results=[{k: v for k, v in r.items() if not k.startswith("_")}
                         for r in s["results"]],
            ) for s in scenarios],
            sweep_figure=os.path.basename(sweep_fig) if sweep_fig else None,
        ), f, indent=2)
    log.info("results -> %s", res_path)
    log.info("=" * 70)
    log.info("DONE")


if __name__ == "__main__":
    main()
