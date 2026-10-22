"""2.8 -- multi-path Pathfinder: m.pathfinder() and fit(init_strategy="pathfinder").

References independent of Pathfinder:
  * a conjugate normal model, whose posterior is known exactly;
  * NUTS on a logistic regression (non-Gaussian posterior);
  * NUTS with a standard 1000-iteration warmup, for the initialisation.
"""
import arviz as az
import jax.numpy as jnp
import numpy as np
import pytest

from BayesForge import bf

pytest.importorskip("blackjax")

# Conjugate model: mu_j ~ N(0, 3^2), y_ij ~ N(mu_j, 1), J groups of n.
J, N_PER, PRIOR_SD = 10, 20, 3.0
_rng = np.random.RandomState(0)
MU_TRUE = _rng.normal(0, 2, J)
Y_CONJ = jnp.asarray(MU_TRUE[:, None] + _rng.normal(0, 1, (J, N_PER)))
POST_PREC = 1 / PRIOR_SD**2 + N_PER
POST_MEAN = np.asarray(Y_CONJ.sum(axis=1)) / POST_PREC
POST_SD = 1 / np.sqrt(POST_PREC)

# Logistic regression.
_X = _rng.normal(0, 1, (200, 3))
_P = 1 / (1 + np.exp(-(-0.5 + _X @ np.array([1.0, -0.8, 0.3]))))
X_LOG, Y_LOG = jnp.asarray(_X), jnp.asarray(_rng.binomial(1, _P))


def _session(seed=1):
    return bf(platform="cpu", rand_seed=seed, print_devices_found=False)


def _conjugate(m):
    def model(y):
        with m.dist.plate("groups", J):
            mu = m.dist.normal(0.0, PRIOR_SD, name="mu")
        m.dist.normal(mu[:, None], 1.0, obs=y)
    return model


def _logistic(m):
    def model(X, y):
        a = m.dist.normal(0, 2.5, name="a")
        b = m.dist.normal(0, 2.5, shape=(3,), name="b")
        m.dist.bernoulli(logits=a + X @ b, obs=y)
    return model


def _z_mcse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mcse = lambda x: x.std() / np.sqrt(float(az.ess(x)))
    return (b.mean() - a.mean()) / np.hypot(mcse(a), mcse(b))


def test_pathfinder_matches_the_exact_conjugate_posterior(verify):
    m = _session()
    m.pathfinder(_conjugate(m), obs={"y": Y_CONJ}, num_samples=2000)
    mu = np.asarray(m.posteriors["mu"])                     # (draws, J)
    err = np.abs(mu.mean(0) - POST_MEAN) / POST_SD          # in posterior sd
    ratio = mu.std(0) / POST_SD
    k = m.pathfinder_info["pareto_k"]
    verify.record(quantity="max |mean - exact| / sd (10 means)", reference="exact posterior",
                  criterion="< 0.1", value=err.max(), passed=err.max() < 0.1,
                  ref=np.random.default_rng(0).normal(POST_MEAN[0], POST_SD, 2000), new=mu[:, 0],
                  ref_label="exact (mu[0])", new_label="pathfinder (mu[0])")
    verify.record(quantity="sd ratio, worst of 10", reference="exact posterior",
                  criterion="0.8 - 1.25", value=ratio[np.argmax(np.abs(np.log(ratio)))],
                  passed=bool(np.all((ratio > 0.8) & (ratio < 1.25))))
    verify.record(quantity="Pareto k", reference="-", criterion="< 0.7", value=k, passed=k < 0.7)
    assert err.max() < 0.1 and np.all((ratio > 0.8) & (ratio < 1.25)) and k < 0.7


def test_pathfinder_close_to_nuts_on_logistic_regression(verify):
    """Multi-path Pathfinder with Pareto-smoothed importance resampling:
    single paths sit on the posterior mode, the importance weights move the
    draws to the posterior mean (logistic posteriors are slightly skewed).
    Tolerance: |Δmean| < 0.1 posterior sd and sd ratio in 0.8-1.25.
    (A sign error in the PSIS call once left the means 0.2-0.3 sd off; this
    test catches it.)"""
    ref = _session(2)
    ref.fit(_logistic(ref), obs={"X": X_LOG, "y": Y_LOG}, num_warmup=500, num_samples=1000,
            num_chains=2, progress_bar=False)
    m = _session(3)
    m.pathfinder(_logistic(m), obs={"X": X_LOG, "y": Y_LOG}, num_samples=2000)
    a_ref, b_ref = np.asarray(ref.posteriors["a"]), np.asarray(ref.posteriors["b"])
    a_pf, b_pf = np.asarray(m.posteriors["a"]), np.asarray(m.posteriors["b"])
    pairs = [("a", a_ref, a_pf)] + [(f"b[{i}]", b_ref[:, i], b_pf[:, i]) for i in range(3)]
    for name, r, p in pairs:
        err = abs(p.mean() - r.mean()) / r.std()
        ratio = p.std() / r.std()
        verify.record(quantity=name, reference="NUTS", criterion="|Δmean| / sd < 0.1",
                      value=err, passed=err < 0.1, ref=r, new=p,
                      ref_label="NUTS", new_label="pathfinder")
        verify.record(quantity=f"{name} sd ratio", reference="NUTS", criterion="0.8 - 1.25",
                      value=ratio, passed=0.8 < ratio < 1.25)
        assert err < 0.1 and 0.8 < ratio < 1.25, f"{name}: err={err:.3f}, sd ratio={ratio:.2f}"
    k = m.pathfinder_info["pareto_k"]
    verify.record(quantity="Pareto k", reference="-", criterion="reported", value=k, passed=True)


def test_pathfinder_initialisation_shortens_warmup(verify):
    data = {"X": X_LOG, "y": Y_LOG}
    ref = _session(4)
    ref.fit(_logistic(ref), obs=data, num_warmup=1000, num_samples=1000, num_chains=4,
            progress_bar=False)
    m = _session(5)
    m.fit(_logistic(m), obs=data, init_strategy="pathfinder", num_warmup=150,
          num_samples=1000, num_chains=4, progress_bar=False)
    for key, idx in (("a", None), ("b", 0), ("b", 1), ("b", 2)):
        r = np.asarray(ref.posteriors_by_chain[key])
        p = np.asarray(m.posteriors_by_chain[key])
        if idx is not None:
            r, p = r[..., idx], p[..., idx]
        name = key if idx is None else f"{key}[{idx}]"
        z, rhat = _z_mcse(r, p), float(az.rhat(p))
        verify.record(quantity=f"{name} (150 warmup)", reference="NUTS, 1000 warmup",
                      criterion="|z| < 3", value=z, passed=abs(z) < 3, ref=r, new=p,
                      ref_label="NUTS 1000 warmup", new_label="pathfinder init, 150 warmup")
        verify.record(quantity=f"{name} R-hat", reference="-", criterion="< 1.01",
                      value=rhat, passed=rhat < 1.01)
        assert abs(z) < 3 and rhat < 1.01, f"{name}: z={z:.2f}, rhat={rhat:.3f}"


def test_unknown_init_strategy_string_is_rejected():
    m = _session()
    with pytest.raises(ValueError, match="init_strategy"):
        m.fit(_logistic(m), obs={"X": X_LOG, "y": Y_LOG}, init_strategy="bogus",
              progress_bar=False)
