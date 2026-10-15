"""Automatic names of observed sites: ``obs=y`` names the site ``"y"``.

Before, an unnamed observed site was called ``"x"`` and collided with a
latent parameter called ``x`` ("duplicate site" error)."""
import jax.numpy as jnp
import numpy as np
from numpyro import handlers

from BayesForge import bf

X = jnp.linspace(-1, 1, 25)
Y = 1.0 + 2.0 * X + 0.1 * jnp.cos(9 * X)


def _session():
    return bf(platform="cpu", rand_seed=1, print_devices_found=False)


def _sites(model, **data):
    with handlers.seed(rng_seed=0):
        tr = handlers.trace(model).get_trace(**data)
    return {k: v["is_observed"] for k, v in tr.items() if v["type"] == "sample"}


def test_obs_variable_names_the_site():
    m = _session()

    def model(x, y):
        a = m.dist.normal(0, 5)
        b = m.dist.normal(0, 5)
        s = m.dist.exponential(1)
        m.dist.normal(a + b * x, s, obs=y)

    assert _sites(model, x=X, y=Y) == {"a": False, "b": False, "s": False, "y": True}


def test_latent_called_x_no_longer_collides():
    m = _session()

    def model(t, y):
        x = m.dist.normal(0, 5)            # a latent called x
        s = m.dist.exponential(1)
        m.dist.normal(x + 0 * t, s, obs=y)  # used to be named "x" too

    m.fit(model, obs=dict(t=X, y=Y), num_warmup=100, num_samples=100,
          num_chains=1, progress_bar=False)
    assert set(m.posteriors) == {"x", "s"}
    assert m.diag.get_yrep().shape == (100, Y.shape[0])


def test_expression_explicit_name_and_assignment():
    m = _session()

    def model(x, y):
        s = m.dist.exponential(1)
        m.dist.normal(0, s, obs=y - 1)                 # expression -> "obs"
        m.dist.normal(0, s, obs=x, name="covariate")   # explicit name kept
        y_hat = m.dist.normal(0, s, obs=y)             # assignment wins
        return y_hat

    assert _sites(model, x=X, y=Y) == {"s": False, "obs": True, "covariate": True, "y_hat": True}


def test_posterior_predictive_uses_the_data_name():
    m = _session()

    def model(x, y):
        a = m.dist.normal(0, 5)
        s = m.dist.exponential(1)
        m.dist.normal(a + 0 * x, s, obs=y)

    m.fit(model, obs=dict(x=X, y=Y), num_warmup=100, num_samples=100,
          num_chains=1, progress_bar=False)
    pred = m.sample(remove_obs=True, posterior=True, seed=0)
    assert "y" in pred and np.asarray(pred["y"]).shape[-1] == Y.shape[0]


def test_two_observed_sites_keep_their_names_in_predictions():
    m = _session()

    def model(x, y, z):
        a = m.dist.normal(0, 5)
        s = m.dist.exponential(1)
        m.dist.normal(a + x, s, obs=y)
        m.dist.normal(a - x, s, obs=z)

    m.fit(model, obs=dict(x=X, y=Y, z=-Y), num_warmup=100, num_samples=100,
          num_chains=1, progress_bar=False)
    pred = m.sample(remove_obs=True, posterior=True, seed=0)
    assert {"y", "z"} <= set(pred)


def test_prior_predictive_before_any_fit():
    m = _session()

    def model(x, y):
        a = m.dist.normal(0, 5)
        m.dist.normal(a + 0 * x, 1.0, obs=y)

    with handlers.seed(rng_seed=0):
        tr = handlers.trace(model).get_trace(x=X, y=None)   # y simulated
    assert "y" in tr and not tr["y"]["is_observed"]
