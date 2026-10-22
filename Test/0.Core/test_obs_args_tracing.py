"""2.7 -- the data arguments that feed observed sites are found by tracing
the model, not by parsing its source (build_model_with_Y_None /
find_obs_in_model).

Before, ``inspect.getsource`` + ``ast`` was used: for a model without
available source (defined with exec, in R / Julia, functools.partial, ...)
no observed argument was found, and predicting on new data without the
outcome failed with "missing a required argument: 'y'"."""
import functools

import jax.numpy as jnp
import numpy as np

from BayesForge import bf

X = jnp.linspace(-1, 1, 30)
Y = 1.0 + 2.0 * X + 0.1 * jnp.cos(9 * X)
X_NEW = jnp.linspace(-2, 2, 7)


def _fit(model, **data):
    m = bf(platform="cpu", rand_seed=1, print_devices_found=False)
    m.fit(model(m), obs=data, num_warmup=100, num_samples=100, num_chains=1, progress_bar=False)
    return m


def _source_model(m):
    def model(x, y):
        a = m.dist.normal(0, 5, name="a")
        b = m.dist.normal(0, 5, name="b")
        s = m.dist.exponential(1, name="s")
        m.dist.normal(a + b * x, s, obs=y, name="y")
    return model


def _exec_model(m):
    # No source available: inspect.getsource raises OSError.
    ns = {"m": m}
    exec("def model(x, y):\n"
         "    a = m.dist.normal(0, 5, name='a')\n"
         "    b = m.dist.normal(0, 5, name='b')\n"
         "    s = m.dist.exponential(1, name='s')\n"
         "    m.dist.normal(a + b * x, s, obs=y, name='y')\n", ns)
    return ns["model"]


def _partial_model(m):
    def model(x, y, scale):
        a = m.dist.normal(0, scale, name="a")
        b = m.dist.normal(0, scale, name="b")
        s = m.dist.exponential(1, name="s")
        m.dist.normal(a + b * x, s, obs=y, name="y")
    return functools.partial(model, scale=5.0)


def test_obs_args_found_by_tracing():
    for build in (_source_model, _exec_model, _partial_model):
        m = _fit(build, x=X, y=Y)
        assert m.find_obs_in_model() == ["y"], build.__name__


def test_predict_on_new_data_without_the_outcome():
    for build in (_source_model, _exec_model, _partial_model):
        m = _fit(build, x=X, y=Y)
        pred = m.sample(data={"x": X_NEW}, remove_obs=True, posterior=True, seed=0)
        assert np.asarray(pred["y"]).shape == (100, X_NEW.shape[0]), build.__name__


def test_same_predictions_with_and_without_source():
    a = _fit(_source_model, x=X, y=Y)
    b = _fit(_exec_model, x=X, y=Y)
    np.testing.assert_array_equal(np.asarray(a.posteriors["b"]), np.asarray(b.posteriors["b"]))
    pa = a.sample(data={"x": X_NEW}, seed=3)["y"]
    pb = b.sample(data={"x": X_NEW}, seed=3)["y"]
    np.testing.assert_array_equal(np.asarray(pa), np.asarray(pb))
