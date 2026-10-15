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
