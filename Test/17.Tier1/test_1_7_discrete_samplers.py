"""Tier 1.7 verification: expose MixedHMC / DiscreteHMCGibbs / HMCGibbs.

Roadmap reference (feature_roadmap.md, Tier 1):
    `m.fit(sampler="nuts"|"mixed_hmc"|"discrete_gibbs"|"hmc_gibbs")`,
    mapped onto `numpyro.infer`.

Verification (per roadmap):
    * N-mixture with small N: posterior of λ and p with `discrete_gibbs` =
      posterior with exact enumeration (NUTS), within MC error.
    * DPMM `latent`: recovers 3 simulated clusters (ARI > 0.9).

Plus smoke coverage that every new sampler name runs end-to-end and that
bad sampler arguments fail fast with a clear error.

Run:
    python Test/17.Tier1/test_1_7_discrete_samplers.py
    (also pytest-collectable)
"""

import functools

import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp
import numpy as np
import numpyro

from BayesForge import bf

# --------------------------------------------------------------------------
# N-mixture data: S sites x J visits, abundance N in 0..NMAX.
#
# The abundance prior is a Poisson(λ) truncated to 0..NMAX, expressed as a
# Categorical over that support: finite (so the discrete latents can be
# summed out exactly) yet parameterised by the literal Poisson rate λ, with
# J = 5 repeated visits making (λ, p) identifiable.
#
# Layout note: the J visits enter as the Binomial trial multiplicity --
# Y_i = Σ_j y_ij ~ Binomial(J·N_i, p), the sufficient statistic of the
# (S, J) visit layout -- under a single `sites` plate. A literal (S, J)
# nested-plate/event layout is not used because funsor cannot reconcile
# the plate dims with the enumeration dim there (broadcast of the plate
# dim against the enum dim); the statistical comparison performed here
# (Gibbs vs exact marginalisation of the same discrete latents) is
# unaffected by the sufficient-statistic reduction.
# --------------------------------------------------------------------------
_NMIX = dict(S=12, J=5, NMAX=8, LAM_TRUE=4.0, P_TRUE=0.6)
_rng_nmix = np.random.RandomState(1)
_N_TRUE = _rng_nmix.poisson(_NMIX["LAM_TRUE"], size=_NMIX["S"]).clip(0, _NMIX["NMAX"])
_Y_NMIX = jnp.asarray(
    np.stack(
        [_rng_nmix.binomial(n, _NMIX["P_TRUE"], size=_NMIX["J"]) for n in _N_TRUE]
    ).sum(axis=1)
)
# Per-site feasible abundance init (J·N >= total count at that site).
_N_INIT = ((np.asarray(_Y_NMIX) + _NMIX["J"] - 1) // _NMIX["J"]).clip(0, _NMIX["NMAX"])
_K_NMIX = jnp.arange(_NMIX["NMAX"] + 1)


def make_nmix(m, enumerate_N):
    """Binomial N-mixture with truncated-Poisson(λ) abundance. `enumerate_N`
    selects exact marginalisation of the discrete latents (reference, fit
    with NUTS) or plain discrete sites (challenger, fit with
    `discrete_gibbs`)."""

    def model(y):
        lam = m.dist.exponential(0.5, name="lam")
        p = m.dist.beta(2.0, 2.0, name="p")
        log_pi = _K_NMIX * jnp.log(lam) - lam - gammaln(_K_NMIX + 1.0)
        pi = jnp.exp(log_pi - logsumexp(log_pi))
        with m.dist.plate("sites", _NMIX["S"]):
            if enumerate_N:
                N = m.dist.categorical_probs(
                    pi, name="N", infer={"enumerate": "parallel"}
                )
            else:
                N = m.dist.categorical_probs(pi, name="N")
            m.dist.binomial_probs(p, total_count=_NMIX["J"] * N, obs=y)

    model.__name__ = "nmix"
    return model


def _new_session(seed=7):
    return bf(platform="cpu", rand_seed=seed, print_devices_found=False)


def test_nmix_discrete_gibbs_matches_enumeration():
    """`discrete_gibbs` posterior of λ and p = exact-enumeration posterior,
    within MC error (|Δmean| < 0.5 pooled sd)."""
    m_ref = _new_session()
    m_ref.fit(
        make_nmix(m_ref, enumerate_N=True),
        obs={"y": _Y_NMIX},
        # A uniform draw can land below an observed count, where the Binomial
        # log-density is -inf: start the abundances at a feasible point.
        init_strategy=numpyro.infer.init_to_value(values={"N": _N_INIT}),
        num_warmup=500,
        num_samples=500,
        num_chains=2,
        progress_bar=False,
    )
    assert m_ref.sampler_name == "nuts"

    m_gibbs = _new_session()
    m_gibbs.fit(
        make_nmix(m_gibbs, enumerate_N=False),
        obs={"y": _Y_NMIX},
        sampler="discrete_gibbs",
        init_strategy=numpyro.infer.init_to_value(values={"N": _N_INIT}),
        # Gibbs conditions each block's init on the other blocks' starting
        # values: start the abundances at a feasible point (N >= y), with one
        # leading entry per chain.
        init_params={"N": np.stack([_N_INIT] * 2)},
        num_warmup=500,
        num_samples=500,
        num_chains=2,
        progress_bar=False,
    )
    assert m_gibbs.sampler_name == "discrete_gibbs"
    # The challenger samples the discrete latents; the reference sums them out.
    assert "N" in m_gibbs.posteriors and "N" not in m_ref.posteriors

    for key in ("lam", "p"):
        a = np.asarray(m_ref.posteriors[key], dtype=np.float64).ravel()
        b = np.asarray(m_gibbs.posteriors[key], dtype=np.float64).ravel()
        pooled_sd = 0.5 * (a.std() + b.std())
        delta = abs(a.mean() - b.mean())
        assert delta < 0.5 * pooled_sd, (
            f"{key}: |Δmean|={delta:.4f} exceeds 0.5*pooled_sd={0.5 * pooled_sd:.4f} "
            f"(enum mean={a.mean():.3f}, gibbs mean={b.mean():.3f})"
        )
        print(f"  {key}: enum mean={a.mean():.3f} sd={a.std():.3f} | "
              f"gibbs mean={b.mean():.3f} sd={b.std():.3f}")
    print("test_nmix_discrete_gibbs_matches_enumeration: OK")


# --------------------------------------------------------------------------
# DPMM latent: 3 well-separated 2-D clusters, truncation T = 3.
# --------------------------------------------------------------------------
_DPMM = dict(N_PER=20, T=3, SEED=11)
_rng_dpmm = np.random.RandomState(2)
_CENTERS = np.array([[-4.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
_X_DPMM = np.vstack(
    [_rng_dpmm.normal(c, 0.7, size=(_DPMM["N_PER"], 2)) for c in _CENTERS]
)
_Z_DPMM = np.repeat([0, 1, 2], _DPMM["N_PER"])


def _ari(true, pred):
    """Adjusted Rand Index (no sklearn dependency)."""
    from math import comb

    true = np.asarray(true)
    pred = np.asarray(pred)
    classes, _ = np.unique(true, return_inverse=True)
    clusters, _ = np.unique(pred, return_inverse=True)
    cont = np.zeros((classes.size, clusters.size), dtype=np.int64)
    np.add.at(cont, (np.searchsorted(np.unique(true), true),
                     np.searchsorted(np.unique(pred), pred)), 1)
    sum_comb = sum(comb(n, 2) for n in cont.ravel())
    sum_rows = sum(comb(n, 2) for n in cont.sum(axis=1))
    sum_cols = sum(comb(n, 2) for n in cont.sum(axis=0))
    n = true.size
    expected = sum_rows * sum_cols / comb(n, 2)
    denom = 0.5 * (sum_rows + sum_cols) - expected
    return (sum_comb - expected) / denom if denom else 0.0


def test_dpmm_latent_recovers_clusters():
    """DPMM `latent` fit with `discrete_gibbs` recovers 3 clusters (ARI > 0.9)."""
    m = _new_session(seed=_DPMM["SEED"])
    m.fit(
        functools.partial(m.models.dpmm, method="latent"),
        obs={"data": jnp.asarray(_X_DPMM), "T": _DPMM["T"]},
        sampler="discrete_gibbs",
        num_warmup=300,
        num_samples=300,
        num_chains=2,
        progress_bar=False,
    )
    z = np.asarray(m.posteriors["z"]).reshape(-1, _Z_DPMM.size)
    aris = [_ari(_Z_DPMM, zd) for zd in z]
    mean_ari = float(np.mean(aris))
    assert mean_ari > 0.9, f"mean ARI={mean_ari:.3f} (per-draw: {np.round(aris, 3)})"
    print(f"test_dpmm_latent_recovers_clusters: OK (mean ARI={mean_ari:.3f})")


# --------------------------------------------------------------------------
# Smoke: mixed_hmc and hmc_gibbs run end-to-end; bad args fail fast.
# --------------------------------------------------------------------------
_rng_smoke = np.random.RandomState(3)
_Y_SMOKE = jnp.asarray(
    np.where(_rng_smoke.binomial(1, 0.5, size=40),
             _rng_smoke.normal(2.0, 1.0, size=40),
             _rng_smoke.normal(-2.0, 1.0, size=40))
)


def test_mixed_hmc_runs():
    m = _new_session()

    def model(y):
        p = m.dist.beta(2.0, 2.0, name="p")
        with m.dist.plate("components", 2):
            mu = m.dist.normal(0.0, 5.0, name="mu")
        with m.dist.plate("N", y.shape[0]):
            c = m.dist.bernoulli_probs(p, name="c")
            m.dist.normal(mu[c], 1.0, obs=y)

    model.__name__ = "smoke_mixed"
    m.fit(model, obs={"y": _Y_SMOKE}, sampler="mixed_hmc",
          num_warmup=200, num_samples=200, num_chains=1, progress_bar=False)
    assert m.sampler_name == "mixed_hmc"
    assert "c" in m.posteriors  # discrete latent is sampled, not enumerated
    mu = np.asarray(m.posteriors["mu"]).reshape(-1, 2).mean(axis=0)
    lo, hi = sorted(float(x) for x in mu)
    assert abs(lo + 2.0) < 0.75 and abs(hi - 2.0) < 0.75, f"mu={mu}"
    print(f"test_mixed_hmc_runs: OK (mu≈({lo:.2f}, {hi:.2f}))")


_Y_GIBBS = jnp.asarray(_rng_smoke.normal(1.5, 1.0, size=30))


def test_hmc_gibbs_runs():
    m = _new_session()

    def model(y):
        x = m.dist.normal(0.0, 2.0, name="x")
        s = m.dist.exponential(1.0, name="s")
        # NOTE: the obs site must be named explicitly here: an unnamed
        # `dist.*(obs=...)` call defaults to name='x', which would collide
        # with the latent above.
        m.dist.normal(x, s, obs=y, name="obs")

    model.__name__ = "smoke_hmc_gibbs"

    n = _Y_GIBBS.shape[0]
    y_sum = float(_Y_GIBBS.sum())

    def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        # x | y, s is Normal in closed form; s comes from the HMC block.
        import jax
        s = hmc_sites["s"]
        post_var = 1.0 / (0.25 + n / s ** 2)
        post_mean = post_var * y_sum / s ** 2
        return {"x": post_mean + jnp.sqrt(post_var) * jax.random.normal(rng_key)}

    m.fit(model, obs={"y": _Y_GIBBS}, sampler="hmc_gibbs",
          gibbs_fn=gibbs_fn, gibbs_sites=["x"],
          num_warmup=200, num_samples=200, num_chains=1, progress_bar=False)
    assert m.sampler_name == "hmc_gibbs"
    x_mean = float(np.asarray(m.posteriors["x"]).mean())
    # Posterior of x concentrates near the sample mean (~1.5); the exact
    # location moves with s, so check a wide but meaningful window.
    assert 0.5 < x_mean < 2.5, f"x_mean={x_mean}"
    print(f"test_hmc_gibbs_runs: OK (x_mean={x_mean:.2f})")


def test_sampler_args_validated():
    m = _new_session()

    def model(y):
        s = m.dist.exponential(1.0, name="s")
        m.dist.normal(0.0, s, obs=y)

    model.__name__ = "smoke_valid"
    for kwargs, msg in (
        ({"sampler": "bogus"}, "unknown sampler"),
        ({"sampler": "hmc_gibbs"}, "hmc_gibbs without gibbs_fn"),
        ({"sampler": "hmc_gibbs", "gibbs_fn": lambda *a: {}}, "hmc_gibbs without sites"),
    ):
        try:
            m.fit(model, obs={"y": _Y_GIBBS}, progress_bar=False, **kwargs)
        except ValueError:
            print(f"test_sampler_args_validated: OK ({msg} -> ValueError)")
        else:
            raise AssertionError(f"{msg}: expected ValueError")


if __name__ == "__main__":
    test_nmix_discrete_gibbs_matches_enumeration()
    test_dpmm_latent_recovers_clusters()
    test_mixed_hmc_runs()
    test_hmc_gibbs_runs()
    test_sampler_args_validated()
    print("1.7 verification: all tests passed.")
