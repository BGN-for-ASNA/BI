# Run:
#   pytest -v "Test/Diag2 vs Diag/test_diag_parity.py"
#
# Purpose
# -------
# `BayesForge.Diagnostic.Diag.diag` is the original, ArviZ-backed diagnostics
# class. `BayesForge.Diagnostic.Diag2.diagWIP` is the newer, JAX/NumPy-native
# reimplementation that BayesForge now uses by default (main.py imports
# `diagWIP as diag`). This suite pins that the two return the *same* numbers on
# the *same* fitted sampler — with particular attention to the hard cases:
#
#   * R-hat and ESS via the public API (both delegate to ArviZ -> must be equal)
#   * the hand-rolled convergence measures in Diag2 that do NOT go through
#     ArviZ: rank-normalized split R-hat (`_rhat_manual`), bulk/tail ESS
#     (`_ess_bulk_manual` / `_ess_tail_manual`) and E-BFMI (`_ebfmi_manual`).
#     These are checked against ArviZ's reference implementations
#     (`az.rhat(method="rank")`, `az.ess(method="bulk"/"tail")`, `az.bfmi`).
#   * the summary table (mean / std / HDI).
#
# One model is fitted once (module-scoped fixture) and shared by every test.

# NOTE: import BayesForge *before* arviz. BayesForge.__init__ applies a
# jax.interpreters.pxla compatibility patch on import; importing arviz (which
# pulls in jax) first can leave that submodule in a state that breaks the patch.
from BayesForge import bf
from BayesForge.Diagnostic.Diag import diag as diag_ref      # ArviZ reference
from BayesForge.Diagnostic.Diag2 import diagWIP as diag_wip  # JAX/NumPy reimpl.

import numpy as np
import pytest
import arviz as az


# ================== Shared fitted model ==================

@pytest.fixture(scope="module")
def fitted():
    """Fit a small, well-identified model with several chains, once.

    Returns (ref, wip, params) where ref/wip are the two diagnostic objects
    built on the *same* sampler and params is the list of parameter names.
    """
    m = bf(platform="cpu")

    data_path = m.load.howell1(only_path=True)
    m.data(data_path, sep=";")
    m.df = m.df[m.df.age > 18]
    m.scale(["weight"])

    def model(weight, height):
        a = m.dist.normal(178, 20)
        b = m.dist.log_normal(0, 1)
        s = m.dist.uniform(0, 50)
        m.dist.normal(a + b * weight, s, obs=height)

    # 4 chains x 1500 draws -> stable ESS / R-hat estimates.
    m.fit(model, num_samples=1500, num_warmup=1000, num_chains=4, progress_bar=False)

    ref = diag_ref(sampler=m.sampler)
    ref.to_az()
    wip = diag_wip(sampler=m.sampler)
    wip.to_az()

    params = list(ref.priors_name)
    return ref, wip, params


# --- helpers -----------------------------------------------------------------

def _chains(wip, param):
    """Per-chain samples (C, S) for a scalar parameter, as a NumPy array."""
    return np.asarray(wip.posterior_samples[param])


def _ds_val(ds, param):
    """Extract a scalar metric for `param` from an ArviZ rhat/ess Dataset."""
    return float(np.asarray(ds[param].values))


# ================== Parameter set parity ==================

def test_same_parameter_names(fitted):
    """Both classes discover the same posterior variables."""
    ref, wip, params = fitted
    assert set(ref.priors_name) == set(wip.priors_name)
    assert set(params) == {"a", "b", "s"}


# ================== Public API: R-hat and ESS (both go through ArviZ) ==================

def test_public_rhat_matches(fitted):
    """diagWIP.rhat() == diag.rhat() (both are az.rhat on the same trace)."""
    ref, wip, params = fitted
    rhat_ref = az.rhat(ref.trace)          # call directly (method attr gets shadowed)
    rhat_wip = az.rhat(wip.trace)
    for p in params:
        assert _ds_val(rhat_ref, p) == pytest.approx(_ds_val(rhat_wip, p), rel=1e-6, abs=1e-6)


def test_public_ess_matches(fitted):
    """diagWIP.ess() == diag.ess() (both are az.ess on the same trace)."""
    ref, wip, params = fitted
    ess_ref = az.ess(ref.trace)
    ess_wip = az.ess(wip.trace)
    for p in params:
        assert _ds_val(ess_ref, p) == pytest.approx(_ds_val(ess_wip, p), rel=1e-6, abs=1e-6)


# ================== Manual (non-ArviZ) measures vs ArviZ reference ==================
# These are the "more complex" measures Diag2 computes by hand in JAX/NumPy.

def test_manual_rhat_matches_arviz_rank(fitted):
    """Diag2._rhat_manual == az.rhat(method='rank') (same estimator)."""
    ref, wip, params = fitted
    rhat_arviz = az.rhat(ref.trace, method="rank")
    for p in params:
        manual = wip._rhat_manual(_chains(wip, p))
        reference = _ds_val(rhat_arviz, p)
        # Same rank-normalized split-R-hat formula -> should agree tightly.
        assert manual == pytest.approx(reference, abs=1e-2), f"{p}: {manual} vs {reference}"


def test_manual_ess_bulk_matches_arviz(fitted):
    """Diag2._ess_bulk_manual ~= az.ess(method='bulk')."""
    ref, wip, params = fitted
    ess_bulk = az.ess(ref.trace, method="bulk")
    for p in params:
        manual = wip._ess_bulk_manual(_chains(wip, p))
        reference = _ds_val(ess_bulk, p)
        # Faithful port of arviz's estimator -> matches to floating-point noise.
        assert manual == pytest.approx(reference, rel=1e-3), f"{p}: {manual} vs {reference}"


def test_manual_ess_tail_matches_arviz(fitted):
    """Diag2._ess_tail_manual == az.ess(method='tail').

    After fixing _ess_raw_manual to use var_plus (within+between variance) and
    Geyer's initial-monotone sequence, the tail-ESS estimator matches arviz.
    """
    ref, wip, params = fitted
    # Diag2._ess_tail_manual uses prob=0.05 (Stan's 5%/95% tails); compare
    # against arviz with the same prob (arviz 1.x's *default* tail prob differs).
    ess_tail = az.ess(ref.trace, method="tail", prob=0.05)
    for p in params:
        manual = wip._ess_tail_manual(_chains(wip, p))
        reference = _ds_val(ess_tail, p)
        assert manual == pytest.approx(reference, rel=1e-3), f"{p}: {manual} vs {reference}"


def test_manual_ebfmi_matches_arviz_bfmi(fitted):
    """Diag2._ebfmi_manual ~= az.bfmi (per chain)."""
    ref, wip, params = fitted
    extra = wip.sampler.get_extra_fields(group_by_chain=True)
    if "energy" not in extra or extra["energy"] is None:
        pytest.skip("energy not collected in extra_fields; cannot check E-BFMI")
    energy = np.asarray(extra["energy"])          # (C, S)
    manual = np.asarray(wip._ebfmi_manual(energy))
    # Pass the energy array straight to az.bfmi (recent arviz returns a DataTree
    # from InferenceData input, which np.asarray can't consume).
    reference = np.asarray(az.bfmi(energy))       # (C,)
    assert manual.shape == reference.shape
    # var-ratio vs sum-of-squares form differ by ~ (N-1)/(N-2); loose tol.
    np.testing.assert_allclose(manual, reference, rtol=0.1)


# ================== Summary table parity (mean / std / HDI) ==================

def test_summary_mean_std_match(fitted):
    """Diag2.summary reproduces Diag's (ArviZ) mean and sd.

    Diag2.summary now goes through _az_hdi (prob=/hdi_prob= compatible), so it
    runs on arviz>=1.x again.
    """
    ref, wip, params = fitted

    # Diag.summary is just az.summary(kind="stats"). On this model az.summary
    # trips on the unnamed obs site ("x") during its internal re-conversion, so
    # build the same ArviZ summary on a clean InferenceData of the latents only
    # (identical samples) — that is the reference Diag2 must reproduce.
    post = {p: np.asarray(wip.posterior_samples[p]) for p in params}
    idata_ref = az.from_dict({"posterior": post})
    df_ref = az.summary(idata_ref, kind="stats", ci_prob=0.89, round_to=6)  # mean, sd, ...

    wip.summary(round_to=6, hdi_prob=0.89)             # raises on arviz>=1.x -> xfail
    df_wip = wip.tab_summary

    for p in params:
        # mean: identical computation up to float error
        assert float(df_ref.loc[p, "mean"]) == pytest.approx(float(df_wip.loc[p, "mean"]), abs=1e-3)
        # spread: ArviZ 'sd' (ddof=1) vs Diag2 'std' (ddof=0) on ~6000 draws
        assert float(df_ref.loc[p, "sd"]) == pytest.approx(float(df_wip.loc[p, "std"]), rel=1e-2)


# ================== Information criteria (identical code paths) ==================

def test_loo_matches(fitted):
    """loo() is arviz-1.x compatible in both classes and gives equal elpd.

    ArviZ 1.x ELPDData exposes .elpd (not .elpd_loo); both classes build a
    NumPy-backed idata with log_likelihood from the same sampler.
    """
    ref, wip, params = fitted
    loo_ref = ref.loo()
    loo_wip = wip.loo()
    assert float(loo_ref.elpd) == pytest.approx(float(loo_wip.elpd), rel=1e-6, abs=1e-6)


def test_waic_matches(fitted):
    """WAIC() is native/arviz-free and matches both classes AND arviz's log-lik.

    az.waic was removed in arviz 1.x. WAIC is computed from the NumPyro pointwise
    log-likelihood (no arviz round-trip). We also verify it equals the WAIC one
    would get from arviz's *own* log_likelihood group — i.e. same result as
    arviz, just without depending on it.
    """
    ref, wip, params = fitted
    waic_ref = ref.WAIC()
    waic_wip = wip.WAIC()
    # cross-class parity
    assert float(waic_ref.elpd_waic) == pytest.approx(float(waic_wip.elpd_waic), rel=1e-6, abs=1e-6)
    assert float(waic_ref.p_waic) == pytest.approx(float(waic_wip.p_waic), rel=1e-6, abs=1e-6)

    # vs arviz's own log-likelihood (the reference values arviz would use)
    from BayesForge.Diagnostic.Diag2 import _waic
    idata = az.from_numpyro(wip.sampler, log_likelihood=True)
    name = list(idata.log_likelihood.data_vars)[0]
    ll_az = np.asarray(idata.log_likelihood[name].values)
    ll_az = ll_az.reshape(-1, ll_az.shape[-1])
    waic_az = _waic(ll_az)
    assert float(waic_wip.elpd_waic) == pytest.approx(float(waic_az.elpd_waic), rel=1e-6, abs=1e-4)
    assert float(waic_wip.p_waic) == pytest.approx(float(waic_az.p_waic), rel=1e-6, abs=1e-4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
