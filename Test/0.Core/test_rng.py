"""Direct draws (sample=True) follow the session seed (rand_seed)."""
import inspect

import jax.numpy as jnp
import numpy as np

from BayesForge import bf
from BayesForge.Distributions.np_dists import UnifiedDist


def test_no_distribution_hardcodes_a_seed():
    # A fixed default seed would make every draw identical and ignore rand_seed.
    fixed = [name for name, f in inspect.getmembers(UnifiedDist, inspect.isfunction)
             if inspect.signature(f).parameters.get("seed") is not None
             and inspect.signature(f).parameters["seed"].default not in (None, inspect.Parameter.empty)]
    assert fixed == []


def _mixture_draw(m):
    return np.asarray(m.dist.mixture(
        mixing_distribution=m.dist.categorical(probs=jnp.array([0.5, 0.5]), create_obj=True),
        component_distributions=[m.dist.normal(0, 1, create_obj=True), m.dist.normal(5, 1, create_obj=True)],
        sample=True, shape=(20,)))


def test_consecutive_draws_differ_and_session_seed_reproduces():
    m = bf(platform="cpu", rand_seed=3, print_devices_found=False)
    first, second = _mixture_draw(m), _mixture_draw(m)
    assert not np.allclose(first, second)
    again = bf(platform="cpu", rand_seed=3, print_devices_found=False)
    np.testing.assert_allclose(_mixture_draw(again), first)
    other = bf(platform="cpu", rand_seed=4, print_devices_found=False)
    assert not np.allclose(_mixture_draw(other), first)


def _fitted(rand_seed):
    m = bf(platform="cpu", rand_seed=rand_seed, print_devices_found=False)
    x = jnp.linspace(-1, 1, 30)
    y = 1.0 + 2.0 * x + 0.2 * jnp.sin(9 * x)

    def model(x, y):
        a = m.dist.normal(0, 5, name="a")
        b = m.dist.normal(0, 5, name="b")
        s = m.dist.exponential(1, name="s")
        m.dist.normal(a + b * x, s, obs=y)

    m.fit(model, obs=dict(x=x, y=y), progress_bar=False, num_samples=100, num_warmup=100, num_chains=1)
    return m


def test_sample_and_yrep_have_no_hidden_fixed_seed():
    m = _fitted(5)
    # no seed: consecutive calls give new draws
    assert not np.allclose(m.diag.get_yrep(), m.diag.get_yrep())
    # explicit seed: identical draws
    np.testing.assert_array_equal(m.diag.get_yrep(seed=1), m.diag.get_yrep(seed=1))
    # same session seed: the whole sequence reproduces
    np.testing.assert_array_equal(_fitted(5).diag.get_yrep(), _fitted(5).diag.get_yrep())


def test_direct_draw_inside_fitted_model_raises_clear_error():
    # sample=True inside a model under JAX tracing must not leak a traced key
    # into the RNG stream (cryptic UnexpectedTracerError); it raises instead.
    import pytest

    m = bf(platform="cpu", rand_seed=1, print_devices_found=False)
    y = jnp.zeros(10)

    def model(y):
        a = m.dist.normal(0, 1, sample=True)
        m.dist.normal(a, 1, obs=y)

    with pytest.raises(RuntimeError, match="sample=True"):
        m.fit(model, obs=dict(y=y), progress_bar=False, num_samples=10, num_warmup=10, num_chains=1)
    # the stream is left untouched and still usable for direct draws
    assert np.isfinite(np.asarray(m.dist.normal(0, 1, sample=True)))
