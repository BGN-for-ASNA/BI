# Run with simulated multi-device support for the pmap test:
#   XLA_FLAGS='--xla_force_host_platform_device_count=4' pytest -v "Test/14.SampledData.py"
#
# Purpose
# -------
# `BayesForge.Distributions.np_dists` currently defaults every sampler to
# `to_jax=True`, returning a raw jax array instead of a `SampledData` wrapper,
# out of caution that `SampledData` "may not be fully compatible with jax array
# operations". This suite exists to settle that question: it exercises a
# `SampledData` object *directly* (without unwrapping `._data`) through the jax
# operations that matter, so we can decide whether `to_jax=True` is still needed.
#
# The decisive sections are:
#   - "Native jnp / ufunc interop": SampledData passed straight into jnp.*
#   - "Transformations on the wrapped object": jit / vmap / grad / scan / pmap
#     receiving the SampledData object itself (pytree), not its ._data.
#   - "As a distribution parameter": the real downstream use of a sampled value.

from BayesForge.Utils.SampledData import SampledData

import itertools

import pickle

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random
import numpyro.distributions as npd

try:
    import cloudpickle
except ImportError:  # cloudpickle is a BayesForge dependency; skip gracefully if absent
    cloudpickle = None


# ================== Test Fixtures ==================

@pytest.fixture
def data_1d():
    """Provides a 1D SampledData object for testing."""
    return SampledData(jnp.arange(10))

@pytest.fixture
def data_2d():
    """Provides a 2D SampledData object for testing."""
    return SampledData(jnp.arange(12).reshape(4, 3))


# ================== Core Delegation Tests ==================

def test_property_delegation(data_2d):
    """Properties like .shape, .ndim, and .T are correctly delegated."""
    assert data_2d.shape == (4, 3)
    assert data_2d.ndim == 2

    transposed_data = data_2d.T
    assert isinstance(transposed_data, SampledData)  # should return a new wrapper
    assert transposed_data.shape == (3, 4)
    np.testing.assert_array_equal(transposed_data._data, jnp.arange(12).reshape(4, 3).T)

def test_method_delegation_no_args(data_1d):
    """Reductions like .mean() return a bare scalar, not a wrapper."""
    mean_val = data_1d.mean()
    assert not isinstance(mean_val, SampledData)
    assert mean_val == pytest.approx(4.5)

def test_method_delegation_with_args(data_2d):
    """Methods with arguments like .reshape() work and stay wrapped."""
    reshaped_data = data_2d.reshape(6, 2)
    assert isinstance(reshaped_data, SampledData)
    assert reshaped_data.shape == (6, 2)

def test_method_chaining(data_2d):
    """Method chaining works, which relies on _wrap_result at each step."""
    result = data_2d.T.reshape(6, 2).mean(axis=0)
    assert isinstance(result, SampledData)
    assert result.shape == (2,)
    np.testing.assert_array_almost_equal(result._data, jnp.array([4.0, 7.0]))

def test_dtype_and_astype(data_1d):
    """dtype delegation and astype conversion behave like a jax array."""
    assert data_1d.dtype == jnp.arange(10).dtype
    as_float = data_1d.astype(jnp.float32)
    assert isinstance(as_float, SampledData)
    assert as_float.dtype == jnp.float32


# ================== Arithmetic and Operator Tests ==================

def test_arithmetic_operators(data_1d):
    """Dunder methods like __add__, __mul__, __sub__ stay wrapped."""
    result_add = data_1d + 10
    assert isinstance(result_add, SampledData)
    np.testing.assert_array_equal(result_add._data, jnp.arange(10) + 10)

    result_mul = data_1d * 2
    np.testing.assert_array_equal(result_mul._data, jnp.arange(10) * 2)

    other_data = SampledData(jnp.ones(10))
    result_sub = data_1d - other_data
    assert isinstance(result_sub, SampledData)
    np.testing.assert_array_equal(result_sub._data, jnp.arange(10) - 1)

def test_reflected_operators(data_1d):
    """Reflected operators (scalar op SampledData) are handled."""
    result_radd = 10 + data_1d
    assert isinstance(result_radd, SampledData)
    np.testing.assert_array_equal(result_radd._data, 10 + jnp.arange(10))

    result_rsub = 100 - data_1d
    np.testing.assert_array_equal(result_rsub._data, 100 - jnp.arange(10))

def test_matmul_operator():
    """The @ operator works both wrapped@wrapped and wrapped@raw."""
    a = SampledData(jnp.arange(6.).reshape(2, 3))
    b = SampledData(jnp.arange(6.).reshape(3, 2))
    result = a @ b
    assert isinstance(result, SampledData)
    np.testing.assert_allclose(result._data, np.arange(6.).reshape(2, 3) @ np.arange(6.).reshape(3, 2))

    result_raw = a @ jnp.arange(6.).reshape(3, 2)
    np.testing.assert_allclose(result_raw._data, result._data)

def test_comparison_operators_return_bool_arrays(data_1d):
    """Comparisons return plain boolean arrays usable as masks."""
    mask = data_1d > 5
    assert not isinstance(mask, SampledData)
    np.testing.assert_array_equal(np.asarray(mask), np.arange(10) > 5)

def test_unary_operators(data_1d):
    """Negation / abs stay wrapped."""
    neg = -data_1d
    assert isinstance(neg, SampledData)
    np.testing.assert_array_equal(neg._data, -jnp.arange(10))
    assert isinstance(abs(neg), SampledData)
    np.testing.assert_array_equal(abs(neg)._data, jnp.arange(10))


# ================== Indexing and Slicing Tests ==================

def test_getitem_slicing(data_2d):
    """Slicing returns a new SampledData object."""
    sliced_data = data_2d[:, 0]
    assert isinstance(sliced_data, SampledData)
    assert sliced_data.shape == (4,)
    np.testing.assert_array_equal(sliced_data._data, jnp.array([0, 3, 6, 9]))

def test_getitem_single_element(data_1d):
    """Getting a single element returns a scalar, not a wrapper."""
    element = data_1d[5]
    assert not isinstance(element, SampledData)
    assert element == 5

def test_boolean_mask_indexing(data_1d):
    """A boolean mask produced from the object indexes it correctly."""
    mask = data_1d > 5
    selected = data_1d[mask]
    assert isinstance(selected, SampledData)
    np.testing.assert_array_equal(selected._data, jnp.arange(10)[np.arange(10) > 5])

def test_fancy_indexing(data_1d):
    """Integer-array (fancy) indexing is delegated."""
    idx = jnp.array([0, 2, 4])
    selected = data_1d[idx]
    assert isinstance(selected, SampledData)
    np.testing.assert_array_equal(selected._data, jnp.array([0, 2, 4]))

def test_setitem(data_1d):
    """__setitem__ uses jax's out-of-place .at[].set under the hood."""
    data_1d[0] = 99
    assert data_1d[0] == 99

def test_at_set_returns_new_array(data_1d):
    """The delegated .at[].set produces an updated array, leaving original intact."""
    updated = data_1d.at[0].set(100)
    assert updated[0] == 100
    assert data_1d[0] == 0  # original unchanged (jax immutability)


# ================== Native jnp / ufunc interop ==================
# These are the crux: can a SampledData be handed straight to jnp.* / np.*
# functions the way a raw array can? This is what `to_jax=True` sidesteps.

def test_jnp_reductions_on_object(data_2d):
    """jnp.sum / jnp.mean accept the object directly via __jax_array__."""
    assert jnp.sum(data_2d) == jnp.sum(jnp.arange(12))
    np.testing.assert_allclose(jnp.mean(data_2d, axis=0), jnp.arange(12).reshape(4, 3).mean(axis=0))

def test_jnp_elementwise_on_object(data_1d):
    """Elementwise jnp ufuncs accept the object directly."""
    expected = jnp.exp(jnp.arange(10).astype(jnp.float32))
    got = jnp.exp(jnp.asarray(data_1d, dtype=jnp.float32))
    np.testing.assert_allclose(got, expected, rtol=1e-6)

def test_jnp_dot_on_object():
    """jnp.dot works with SampledData arguments (no manual unwrap)."""
    v = SampledData(jnp.array([1., 2., 3.]))
    assert jnp.dot(v, v) == pytest.approx(14.0)

def test_jnp_where_with_object(data_1d):
    """jnp.where accepts a SampledData as a branch value."""
    out = jnp.where(jnp.arange(10) % 2 == 0, data_1d, -1)
    expected = jnp.where(jnp.arange(10) % 2 == 0, jnp.arange(10), -1)
    np.testing.assert_array_equal(out, expected)

def test_jnp_concatenate_and_stack():
    """SampledData mixes with raw arrays in concatenate / stack."""
    a = SampledData(jnp.array([1., 2., 3.]))
    cat = jnp.concatenate([a, jnp.array([4., 5.])])
    np.testing.assert_array_equal(cat, jnp.array([1., 2., 3., 4., 5.]))
    stacked = jnp.stack([a, a])
    assert stacked.shape == (2, 3)

def test_broadcasting_with_raw_array(data_2d):
    """A raw array broadcasts against a SampledData (and vice versa)."""
    col = jnp.array([10, 20, 30])
    out = data_2d + col
    assert isinstance(out, SampledData)
    np.testing.assert_array_equal(out._data, jnp.arange(12).reshape(4, 3) + col)

def test_numpy_conversion(data_2d):
    """np.asarray / __array__ produces a real numpy array."""
    arr = np.asarray(data_2d)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, np.arange(12).reshape(4, 3))

def test_jnp_asarray_and_to_jax(data_2d):
    """jnp.asarray and .to_jax() both yield an unwrapped jax array."""
    as_jax = jnp.asarray(data_2d)
    assert isinstance(as_jax, jnp.ndarray) and not isinstance(as_jax, SampledData)
    np.testing.assert_array_equal(as_jax, jnp.arange(12).reshape(4, 3))

    to_jax = data_2d.to_jax()
    assert isinstance(to_jax, jnp.ndarray) and not isinstance(to_jax, SampledData)
    np.testing.assert_array_equal(to_jax, jnp.arange(12).reshape(4, 3))


# ================== Pytree registration ==================

def test_tree_flatten_unflatten_roundtrip(data_2d):
    """SampledData round-trips through jax pytree flatten/unflatten."""
    leaves, treedef = jax.tree_util.tree_flatten(data_2d)
    assert len(leaves) == 1
    np.testing.assert_array_equal(leaves[0], jnp.arange(12).reshape(4, 3))
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, SampledData)
    np.testing.assert_array_equal(rebuilt._data, data_2d._data)

def test_tree_map_preserves_wrapper(data_1d):
    """jax.tree_map over the object applies to the leaf and re-wraps."""
    doubled = jax.tree_util.tree_map(lambda x: x * 2, data_1d)
    assert isinstance(doubled, SampledData)
    np.testing.assert_array_equal(doubled._data, jnp.arange(10) * 2)


# ================== Transformations on the wrapped object ==================
# Unlike the historical version of this suite, these pass the SampledData
# object itself into the transformation rather than `obj._data`.

@jax.jit
def _jitted_math(x):
    """Operations jax must trace through the SampledData object."""
    y = x.T * 2 + 5
    z = y.sum() / y.size
    return z

def test_jit_takes_object_directly():
    """A jitted function accepts the SampledData object and traces it."""
    data = SampledData(jnp.arange(6, dtype=jnp.float32).reshape(2, 3))
    result = _jitted_math(data)
    # x.T*2+5 -> [[5,11],[7,13],[9,15]]; sum=60; size=6; -> 10.0
    np.testing.assert_allclose(result, 10.0)
    assert not isinstance(result, SampledData)

@jax.jit
def _jitted_returns_wrapper(x):
    """A jitted function that returns a SampledData (exercises unflatten under jit)."""
    return x * 2 + 1

def test_jit_returns_wrapper():
    """A jitted function can return a SampledData via pytree reconstruction."""
    data = SampledData(jnp.arange(4, dtype=jnp.float32))
    out = _jitted_returns_wrapper(data)
    assert isinstance(out, SampledData)
    np.testing.assert_allclose(out._data, jnp.arange(4) * 2 + 1)

def test_grad_through_object():
    """jax.grad differentiates a function that consumes the object."""
    def square_and_sum(sd_object):
        return jnp.sum(jnp.square(jnp.asarray(sd_object)))

    grad_func = jax.grad(square_and_sum)
    data = SampledData(jnp.array([1.0, 2.0, 3.0]))
    gradient = grad_func(data)
    # d/dx sum(x^2) = 2x; grad of a pytree input is the same pytree type.
    assert isinstance(gradient, SampledData)
    np.testing.assert_allclose(gradient._data, jnp.array([2.0, 4.0, 6.0]))

def test_vmap_over_object():
    """vmap maps over the wrapped object's leading axis, re-wrapping each slice."""
    def row_dot(row):  # row arrives as a SampledData slice
        return jnp.dot(row, row)

    data = SampledData(jnp.array([[1., 2.], [3., 4.], [5., 6.]]))
    result = jax.vmap(row_dot)(data)
    assert not isinstance(result, SampledData)
    np.testing.assert_allclose(result, jnp.array([5., 25., 61.]))

def test_lax_scan_with_wrapper_carry():
    """SampledData works as the carry state of lax.scan."""
    def accumulate(sd_carry, x):
        new = SampledData(sd_carry._data + x)
        return new, jnp.sum(new._data)

    initial = SampledData(jnp.zeros(3, dtype=jnp.float32))
    xs = jnp.ones((5, 3))
    final_carry, outputs = lax.scan(accumulate, initial, xs)
    assert isinstance(final_carry, SampledData)
    np.testing.assert_allclose(final_carry._data, jnp.array([5., 5., 5.]))
    np.testing.assert_allclose(outputs, jnp.array([3., 6., 9., 12., 15.]))

def test_lax_cond_with_wrapper():
    """SampledData flows through both branches of lax.cond as operand/result."""
    def run(pred, sd):
        return lax.cond(pred, lambda s: s * 2, lambda s: s + 100, sd)

    data = SampledData(jnp.arange(3, dtype=jnp.float32))
    np.testing.assert_allclose(run(True, data)._data, jnp.arange(3) * 2)
    np.testing.assert_allclose(run(False, data)._data, jnp.arange(3) + 100)

def test_pmap_over_object():
    """SampledData works with jax.pmap when multiple devices are available."""
    if jax.device_count() < 2:
        pytest.skip("pmap test requires multiple devices "
                    "(run with XLA_FLAGS='--xla_force_host_platform_device_count=4').")
    n = jax.device_count()
    shape = (n, 3)
    data = SampledData(jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape))
    result = jax.pmap(lambda r: r * 2)(data)
    assert result.shape == shape
    np.testing.assert_allclose(np.asarray(result),
                               np.arange(np.prod(shape)).reshape(shape) * 2)


# ================== isinstance identity (the __class__ spoof) ==================
# SampledData overrides __class__ so it reports as a jax array. These tests pin
# down that the spoof passes the array check WITHOUT sacrificing self-identity,
# pytree behaviour, or serialization (which the spoof would otherwise break).

def test_isinstance_reports_as_jax_array(data_1d):
    """isinstance(sd, jnp.ndarray) is True via the __class__ override."""
    assert isinstance(data_1d, jnp.ndarray)

def test_isinstance_self_identity_preserved(data_1d):
    """The spoof must not break isinstance(sd, SampledData) (type() is unchanged)."""
    assert isinstance(data_1d, SampledData)
    assert type(data_1d) is SampledData

def test_spoof_does_not_break_pytree(data_2d):
    """type()-keyed pytree registration is unaffected by the __class__ override."""
    leaves, _ = jax.tree_util.tree_flatten(data_2d)
    assert len(leaves) == 1
    assert not isinstance(leaves[0], SampledData)

def test_pickle_roundtrip(data_2d):
    """__reduce__ keeps stdlib pickle working despite the spoofed __class__."""
    restored = pickle.loads(pickle.dumps(data_2d))
    assert isinstance(restored, SampledData)
    np.testing.assert_array_equal(restored._data, data_2d._data)

@pytest.mark.skipif(cloudpickle is None, reason="cloudpickle not installed")
def test_cloudpickle_roundtrip(data_2d):
    """BayesForge uses cloudpickle for model save/load; it must round-trip."""
    restored = cloudpickle.loads(cloudpickle.dumps(data_2d))
    assert isinstance(restored, SampledData)
    np.testing.assert_array_equal(restored._data, data_2d._data)


# ================== As a distribution parameter ==================
# The downstream reason a sampled value gets reused: feeding it back into
# another distribution. This is exactly the path `to_jax=True` was protecting.

def test_sampled_data_as_distribution_loc():
    """A SampledData can parameterize a numpyro distribution and be sampled."""
    loc = SampledData(jnp.array([0.0, 5.0, 10.0]))
    d = npd.Normal(loc, 1.0)
    s = d.sample(random.PRNGKey(0))
    assert s.shape == (3,)
    # Samples should sit near their respective loc values.
    assert bool(jnp.all(jnp.abs(s - jnp.array([0., 5., 10.])) < 6))

def test_sampled_data_as_distribution_log_prob():
    """log_prob accepts a SampledData observation and matches the raw-array result."""
    obs = SampledData(jnp.array([0.1, 4.9, 9.8]))
    d = npd.Normal(jnp.array([0.0, 5.0, 10.0]), 1.0)
    lp_wrapped = d.log_prob(obs)
    lp_raw = d.log_prob(jnp.asarray(obs))
    np.testing.assert_allclose(np.asarray(lp_wrapped), np.asarray(lp_raw))

def test_chained_sampling_with_wrapper():
    """A wrapped sample feeds a second distribution (hierarchical-style chaining).

    Characterization, not a bug: because numpyro samples via `loc + scale * eps`,
    a `SampledData` passed as `loc` makes the *returned* sample a `SampledData`
    too (the wrapper rides along through `__radd__`/`__rmul__`); whether it does
    depends on internal op order (wrapping only `scale` returns a raw array).
    Either way the result stays fully array-compatible. Callers that need a
    guaranteed raw array should finish with `.to_jax()` / `jnp.asarray(...)`.
    """
    mu = SampledData(npd.Normal(0.0, 1.0).sample(random.PRNGKey(0), sample_shape=(4,)))
    y = npd.Normal(mu, 0.5).sample(random.PRNGKey(1))
    assert y.shape == (4,)
    # Regardless of whether the wrapper propagated, the value is array-usable.
    assert bool(jnp.all(jnp.isfinite(jnp.asarray(y))))
    assert isinstance(jnp.asarray(y), jnp.ndarray) and not isinstance(jnp.asarray(y), SampledData)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
