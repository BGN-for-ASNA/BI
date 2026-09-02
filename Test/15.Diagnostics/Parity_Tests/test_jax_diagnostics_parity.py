"""ArviZ parity tests for BayesForge.Diagnostic.jax_diagnostics.

This is the module `main.py` actually uses: `m.summary()` calls
`jax_diagnostics.summary`, and `patch_diag.bind_diag_to_model` routes
`m.diag.rhat/ess/mcse/loo/WAIC` here. Before this file existed the only
accuracy test in the repo (`Test/Diag2 vs Diag/test_diag_parity.py`) tested
`Diag2`'s *manual* estimators and never touched this module at all, which is
how the following shipped:

  * r_hat was plain split R-hat, not the rank-normalized folded-max estimator
    the 1.01 threshold is calibrated for -> 10.75% under-report
  * mcse_sd used a normal approximation instead of ArviZ's delta-method form
  * mcse_mean divided by ess_bulk instead of ess_mean
  * filter_posterior_dict never matched keys carrying an index suffix
  * ess()/mcse() raised IndexError on the flat posteriors m.posteriors holds

ArviZ is the oracle throughout. Tolerances are tight on purpose: a tolerance
wide enough to absorb a wrong estimator is not a test.
"""
import numpy as np
import pytest

import arviz as az

from BayesForge.Diagnostic import jax_diagnostics as jd


# ============================== fixtures ==============================

@pytest.fixture(scope="module")
def chains():
    """Four chain sets, each stressing a different failure mode.

    Returns (dict_of_arrays, InferenceData) over the same draws.
    """
    rng = np.random.default_rng(0)

    def ar1(rho, n, sd=1.0):
        x = np.zeros(n)
        e = rng.normal(0, sd, n)
        for i in range(1, n):
            x[i] = rho * x[i - 1] + e[i]
        return x

    C, S = 4, 2000
    post = {
        # well mixed
        "well":  np.stack([ar1(0.6, S) for _ in range(C)]),
        # chain MEANS disagree -> caught by plain split R-hat too
        "bad":   np.stack([ar1(0.9, S) + 0.4 * c for c in range(C)]),
        # chain VARIANCES disagree -> ONLY the folded half of rank-normalized
        # R-hat sees this. Plain split R-hat reports 1.0006 where ArviZ says
        # 1.1211, i.e. "converged" for a model that is not.
        "scale": np.stack([ar1(0.5, S, sd=1 + 0.8 * c) for c in range(C)]),
        # heavy right skew -> separates rank-normalized from raw estimators
        "skew":  np.stack([np.exp(ar1(0.7, S)) for _ in range(C)]),
    }
    # NOTE: keyword, not a positional dict. az.from_dict's first parameter IS
    # `posterior`, so from_dict({"posterior": ...}) nests it a level too deep.
    return post, az.from_dict(posterior=post)


PARAMS = ["well", "bad", "scale", "skew"]


# ============================== R-hat ==============================

@pytest.mark.parametrize("param", PARAMS)
def test_rhat_matches_arviz_rank(chains, param):
    post, idata = chains
    expected = float(az.rhat(idata, method="rank")[param].values)
    assert float(jd._rhat_1d(post[param])) == pytest.approx(expected, rel=1e-6)


def test_rhat_detects_variance_nonconvergence(chains):
    """Regression guard for the estimator swap, stated as a behaviour.

    The `scale` chains differ only in variance. Plain split R-hat scores them
    ~1.0006 (below the 1.01 threshold, i.e. a silent false negative); the
    correct estimator scores them well above it.
    """
    post, _ = chains
    assert float(jd._rhat_1d(post["scale"])) > 1.05


def test_rhat_dict_api(chains):
    post, idata = chains
    got = jd.rhat(post)
    for p in PARAMS:
        assert got[p] == pytest.approx(
            float(az.rhat(idata, method="rank")[p].values), rel=1e-6)


# ============================== ESS ==============================

@pytest.mark.parametrize("param", PARAMS)
@pytest.mark.parametrize("kind", ["bulk", "tail"])
def test_ess_matches_arviz(chains, param, kind):
    post, idata = chains
    fn = jd._ess_1d if kind == "bulk" else jd._ess_tail_1d
    expected = float(az.ess(idata, method=kind)[param].values)
    assert fn(post[param]) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("param", PARAMS)
@pytest.mark.parametrize("kind", ["mean", "sd"])
def test_ess_mean_and_sd_match_arviz(chains, param, kind):
    post, idata = chains
    fn = jd._ess_mean_1d if kind == "mean" else jd._ess_sd_1d
    expected = float(az.ess(idata, method=kind)[param].values)
    assert fn(post[param]) == pytest.approx(expected, rel=1e-6)


# ============================== summary table ==============================

SUMMARY_COLS = ["mean", "sd", "mcse_mean", "mcse_sd",
                "ess_bulk", "ess_tail", "r_hat", "hdi_5.5%", "hdi_94.5%"]


@pytest.mark.parametrize("col", SUMMARY_COLS)
@pytest.mark.parametrize("param", PARAMS)
def test_summary_column_matches_arviz(chains, col, param):
    post, idata = chains
    got = jd.summary(post, round_to=8, hdi_prob=0.89, group_by_chain=True)
    ref = az.summary(idata, round_to=8, hdi_prob=0.89)
    assert float(got.loc[param, col]) == pytest.approx(
        float(ref.loc[param, col]), rel=1e-6)


def test_summary_reports_rhat_for_a_single_chain():
    """Split R-hat is defined for one chain; it was forced to NaN."""
    x = np.random.default_rng(4).normal(size=(1, 2000))
    df = jd.summary({"a": x}, group_by_chain=True, round_to=8)
    assert np.isfinite(float(df.loc["a", "r_hat"]))


# ============================== HDI ==============================

@pytest.mark.parametrize("n,p", [(1000, 0.89), (997, 0.94), (333, 0.5),
                                 (500, 0.95), (1234, 0.8)])
def test_hdi_matches_arviz(n, p):
    """Fails if hdi() rounds the interval size with ceil instead of floor."""
    x = np.random.default_rng(1).normal(size=n)
    np.testing.assert_allclose(np.asarray(jd.hdi(x, hdi_prob=p)),
                               np.asarray(az.hdi(x, hdi_prob=p)), rtol=1e-12)


def test_hdi_returns_double_precision():
    x = np.random.default_rng(1).normal(size=500)
    assert np.asarray(jd.hdi(x, 0.89)).dtype == np.float64


# ============================== LOO / WAIC ==============================

@pytest.fixture(scope="module")
def loglik(chains):
    post, _ = chains
    rng = np.random.default_rng(2)
    ll = rng.normal(-2.0, 0.6, size=(4, 2000, 60))
    idata = az.from_dict(posterior={"z": post["well"]}, log_likelihood={"y": ll})
    return ll, idata


def test_waic_matches_arviz(loglik):
    ll, idata = loglik
    got, ref = jd.waic(ll), az.waic(idata)
    assert got.elpd == pytest.approx(float(ref.elpd_waic), rel=1e-6)
    assert got.p == pytest.approx(float(ref.p_waic), rel=1e-6)
    assert got.se == pytest.approx(float(ref.se), rel=1e-6)


def test_loo_matches_arviz(loglik):
    ll, idata = loglik
    got, ref = jd.loo(ll, pointwise=True), az.loo(idata, pointwise=True)
    assert got.elpd == pytest.approx(float(ref.elpd_loo), rel=1e-4)
    assert got.p == pytest.approx(float(ref.p_loo), rel=1e-3)


def test_loo_pareto_k_matches_arviz_when_reff_is_supplied(chains, loglik):
    """PSIS sizes its tail as 3*sqrt(S/reff).

    reff comes from the POSTERIOR (that is what az.loo does), not from the
    log-likelihood. Hardcoding reff=1 left pareto_k off by up to 0.13 -- enough
    to move observations across the 0.7 threshold that
    sensitivity.influence_plot colours on.
    """
    post, _ = chains
    ll, idata = loglik
    reff = jd.relative_eff({"z": post["well"]})
    got = jd.loo(ll, pointwise=True, reff=reff)
    ref = az.loo(idata, pointwise=True, reff=reff)
    np.testing.assert_allclose(np.asarray(got.pareto_k),
                               np.asarray(ref.pareto_k.values).ravel(), atol=1e-6)


def test_relative_eff_is_a_fraction(chains):
    post, _ = chains
    reff = jd.relative_eff(post)
    assert 0.0 < reff <= 1.5


@pytest.mark.parametrize("scale,mult", [("log", 1.0), ("negative_log", -1.0),
                                        ("deviance", -2.0)])
def test_waic_scales(loglik, scale, mult):
    ll, _ = loglik
    assert jd.waic(ll, scale=scale).elpd == pytest.approx(
        mult * jd.waic(ll).elpd, rel=1e-9)


# ============================== compare ==============================

def test_compare_elpd_diff_sign_matches_arviz():
    """elpd_diff must be 0 for the best model and POSITIVE for worse ones."""
    rng = np.random.default_rng(5)
    good = rng.normal(-1.0, 0.4, size=(2, 500, 40))
    bad = rng.normal(-3.0, 0.4, size=(2, 500, 40))
    df = jd.compare({"good": good, "bad": bad}, ic="waic")
    assert df.loc["good", "rank"] == 0
    assert df.loc["good", "elpd_diff"] == pytest.approx(0.0, abs=1e-9)
    assert df.loc["bad", "elpd_diff"] > 0


def test_compare_rejects_mismatched_observation_counts():
    rng = np.random.default_rng(6)
    a = rng.normal(size=(2, 200, 40))
    b = rng.normal(size=(2, 200, 30))
    with pytest.raises(ValueError, match="same observations"):
        jd.compare({"a": a, "b": b}, ic="waic")


# ============================== filtering / shapes ==============================

def test_filter_posterior_dict_matches_base_names():
    d = {"mu[0]": np.zeros(3), "mu[1]": np.zeros(3), "sigma": np.zeros(3)}
    assert set(jd.filter_posterior_dict(d, include="mu")) == {"mu[0]", "mu[1]"}
    assert set(jd.filter_posterior_dict(d, exclude="mu")) == {"sigma"}


def test_filter_posterior_dict_does_not_match_prefixes():
    d = {"var": np.zeros(3), "var_1": np.zeros(3)}
    assert set(jd.filter_posterior_dict(d, include="var")) == {"var"}


def test_ess_and_mcse_accept_flat_posteriors():
    """m.posteriors is flat (N,); these used to raise IndexError."""
    flat = {"a": np.random.default_rng(3).normal(size=2000)}
    assert np.isfinite(jd.ess(flat)["a"])
    assert np.isfinite(jd.mcse(flat)["a"])


def test_ess_and_mcse_accept_flat_vector_posteriors():
    """Flat (N, K) draws of a K-vector, declared via group_by_chain=False."""
    flat = {"b": np.random.default_rng(3).normal(size=(2000, 3))}
    assert jd.ess(flat, group_by_chain=False)["b"].shape == (3,)
    assert jd.mcse(flat, group_by_chain=False)["b"].shape == (3,)


@pytest.mark.parametrize("kind", ["mean", "sd"])
def test_mcse_matches_arviz(chains, kind):
    post, idata = chains
    got = jd.mcse(post, kind=kind)
    ref = az.mcse(idata, method=kind)
    for p in PARAMS:
        assert got[p] == pytest.approx(float(ref[p].values), rel=1e-6)


def test_multidim_parameters_are_expanded(chains):
    post, _ = chains
    vec = {"v": np.stack([post["well"], post["bad"]], axis=-1)}  # (C, S, 2)
    assert jd.rhat(vec)["v"].shape == (2,)
    assert jd.ess(vec)["v"].shape == (2,)
    df = jd.summary(vec, group_by_chain=True)
    assert list(df.index) == ["v[0]", "v[1]"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
