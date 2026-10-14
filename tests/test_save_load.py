"""Saving a fitted bf object and loading it back.

Regression test: ``m.save()`` failed with "cannot pickle 'jaxlib._jax.Device'
object" (the device mesh/shardings were pickled) but only printed the error,
so ``bf.load()`` then returned ``None``.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from BayesForge import bf


@pytest.fixture(scope="module")
def fitted():
    m = bf(platform="cpu", rand_seed=42, print_devices_found=False)
    x = jnp.linspace(-1, 1, 40)
    y = 0.5 + 2.0 * x + 0.1 * jnp.cos(7 * x)

    def model(x, y):
        alpha = m.dist.normal(0, 10, name="alpha")
        beta = m.dist.normal(0, 1, name="beta")
        sigma = m.dist.exponential(1, name="sigma")
        m.dist.normal(alpha + beta * x, sigma, obs=y)

    m.fit(model, obs=dict(x=x, y=y), progress_bar=False, num_samples=200, num_warmup=200)
    return m


def test_round_trip(fitted, tmp_path):
    path = fitted.save(str(tmp_path / "lr.pkl"))
    loaded = bf.load(path)
    assert isinstance(loaded, bf)
    np.testing.assert_allclose(fitted.posteriors["beta"], loaded.posteriors["beta"])
    assert np.isfinite(float(loaded.diag.loo().elpd))
    if loaded.n_devices > 1:
        assert loaded.mesh is not None  # rebuilt from the local devices


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        bf.load(str(tmp_path / "missing.pkl"))
