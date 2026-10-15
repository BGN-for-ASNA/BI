"""Information criteria and model comparison on a fitted model.

Regression tests for the MEE-26-09-855 review: ``m.diag.loo()`` raised
``TypeError: from_dict() got an unexpected keyword argument 'posterior'``
under ArviZ >= 1.0.
"""
import numpy as np
import pytest

from BayesForge import bf


def _fit(with_slope):
    m = bf(platform="cpu")
    m.data(m.load.howell1(only_path=True), sep=";")
    m.df = m.df[m.df.age > 18]
    m.scale(["weight"])

    def model(weight, height):
        a = m.dist.normal(178, 20, name="a")
        b = m.dist.log_normal(0, 1, name="b") if with_slope else 0.0
        s = m.dist.uniform(0, 50, name="s")
        m.dist.normal(a + b * weight, s, obs=height)

    m.fit(model, progress_bar=False, num_samples=200, num_warmup=200)
    return m


@pytest.fixture(scope="module")
def slope_model():
    return _fit(with_slope=True)


@pytest.fixture(scope="module")
def intercept_model():
    return _fit(with_slope=False)


def test_loo(slope_model):
    res = slope_model.diag.loo()
    assert np.isfinite(float(res.elpd))


def test_loo_pointwise(slope_model):
    res = slope_model.diag.loo(pointwise=True)
    assert np.asarray(res.elpd_i).size == slope_model.df.shape[0]


def test_waic_and_alias(slope_model):
    lower = slope_model.diag.waic()
    upper = slope_model.diag.WAIC()
    assert np.isfinite(lower["elpd_waic"])
    assert lower["elpd_waic"] == pytest.approx(upper["elpd_waic"])


def test_compare_fitted_models(slope_model, intercept_model):
    table = slope_model.diag.compare({"slope": slope_model, "intercept": intercept_model})
    assert list(table.index) == ["slope", "intercept"]
