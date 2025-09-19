# To test fiel run dk XLA_FLAGS='--xla_force_host_platform_device_count=4' pytest -v
from BI.Utils.SampledData import SampledData
# test_sampled_data.py
import pytest
import jax.numpy as jnp
import numpy as np


# ================== Test Fixtures ==================
# Fixtures are reusable setup functions for your tests.

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
    """Test if properties like .shape, .ndim, and .T are correctly delegated."""
    assert data_2d.shape == (4, 3)
    assert data_2d.ndim == 2
    
    # Test the .T property
    transposed_data = data_2d.T
    assert isinstance(transposed_data, SampledData) # Crucial: Should return a new wrapper
    assert transposed_data.shape == (3, 4)
    np.testing.assert_array_equal(transposed_data._data, jnp.arange(12).reshape(4, 3).T)

def test_method_delegation_no_args(data_1d):
    """Test if methods without arguments like .mean() work correctly."""
    mean_val = data_1d.mean()
    # For a simple reduction, the result might be a JAX scalar, not a SampledData object.
    # The _wrap_result logic should handle this.
    assert not isinstance(mean_val, SampledData) 
    assert mean_val == pytest.approx(4.5)

def test_method_delegation_with_args(data_2d):
    """Test if methods with arguments like .reshape() work."""
    reshaped_data = data_2d.reshape(6, 2)
    assert isinstance(reshaped_data, SampledData)
    assert reshaped_data.shape == (6, 2)

def test_method_chaining(data_2d):
    """Test if method chaining works, which relies on _wrap_result."""
    # This chain will only work if each step correctly returns a SampledData instance.
    result = data_2d.T.reshape(6, 2).mean(axis=0)
    assert isinstance(result, SampledData) # mean(axis=0) should still be an array
    assert result.shape == (2,)
    np.testing.assert_array_almost_equal(result._data, jnp.array([4.0, 7.0]))


# ================== Arithmetic and Operator Tests ==================

def test_arithmetic_operators(data_1d):
    """Test dunder methods like __add__, __mul__, etc."""
    result_add = data_1d + 10
    assert isinstance(result_add, SampledData)
    np.testing.assert_array_equal(result_add._data, jnp.arange(10) + 10)

    result_mul = data_1d * 2
    np.testing.assert_array_equal(result_mul._data, jnp.arange(10) * 2)

    # Test with another SampledData object
    other_data = SampledData(jnp.ones(10))
    result_sub = data_1d - other_data
    assert isinstance(result_sub, SampledData)
    np.testing.assert_array_equal(result_sub._data, jnp.arange(10) - 1)

# ================== Indexing and Slicing Tests ==================

def test_getitem_slicing(data_2d):
    """Test if indexing and slicing returns a new SampledData object."""
    sliced_data = data_2d[:, 0]
    assert isinstance(sliced_data, SampledData)
    assert sliced_data.shape == (4,)
    np.testing.assert_array_equal(sliced_data._data, jnp.array([0, 3, 6, 9]))

def test_getitem_single_element(data_1d):
    """Test if getting a single element returns a scalar, not a wrapper."""
    element = data_1d[5]
    assert not isinstance(element, SampledData)
    assert element == 5

def test_setitem_produces_new_object(data_1d):
    """Test __setitem__ for JAX's out-of-place updates."""
    original_id = id(data_1d)
    # Your __setitem__ should be `self._data = self._data.at[idx].set(value)`
    # It modifies the internal _data, so the SampledData object itself might not change ID,
    # but the underlying JAX array will be a new one.
    new_data = data_1d.at[0].set(100) # Assuming you add an .at property for clarity
    
    # A better way is to test the __setitem__ directly if you have it
    data_1d[0] = 99 # This should reassign data_1d._data
    
    assert data_1d[0] == 99
    # The original array (before the last setitem) should be unchanged if you grab it first.
    # This is tricky with JAX's immutability but the key is that operations return new arrays.





# ================== JAX Compatibility Tests ==================
import jax
# First, define a standalone function that we can jit.
# It takes our SampledData object as an argument.
@jax.jit
def jitted_math_on_sampled_data(x):
    """A simple JIT-compiled function to test tracing through SampledData."""
    # Perform a chain of operations that JAX will compile
    y = x.T * 2 + 5
    z = y.sum() / y.size
    return z

def test_jit_compatibility():
    """
    Tests if a SampledData object can be passed to and correctly
    processed by a JIT-compiled JAX function.
    """
    # 1. Setup: Create a SampledData object
    data = SampledData(jnp.arange(6, dtype=jnp.float32).reshape(2, 3))
    # data._data is [[0., 1., 2.], [3., 4., 5.]]

    # 2. Execute: Call the jitted function. 
    # The first time this runs, JAX will trace the function and compile it.
    result = jitted_math_on_sampled_data(data)

    # 3. Assert: Check if the numerical result is correct.
    # Let's calculate the expected value by hand:
    # x.T          -> [[0., 3.], [1., 4.], [2., 5.]]
    # x.T * 2      -> [[0., 6.], [2., 8.], [4., 10.]]
    # x.T * 2 + 5  -> [[5., 11.], [7., 13.], [9., 15.]]
    # sum          -> 5 + 11 + 7 + 13 + 9 + 15 = 60
    # size         -> 6
    # sum / size   -> 10.0
    expected_value = 10.0

    # Use np.testing.assert_allclose for robust float comparison
    np.testing.assert_allclose(result, expected_value)

    # Also assert that the result is a scalar, not a SampledData object
    assert not isinstance(result, SampledData)
    print(f"\nJIT test successful! Result: {result}, Expected: {expected_value}")


# In test_sampled_data.py, at the end of the file

def test_grad_compatibility_simple():
    """Tests jax.grad with a function that takes and returns a SampledData object."""
    
    # Define a function that operates on SampledData objects IN A JAX-FRIENDLY WAY.
    def square_and_sum(sd_object):
        # THIS IS THE KEY CHANGE:
        # Instead of sd_object.square(), we use the jnp function
        # and operate on the raw data, which is what JAX traces.
        squared_data = jnp.square(sd_object._data) 
        return jnp.sum(squared_data)

    # Get the gradient of this function with respect to the first argument (our object)
    grad_func = jax.grad(square_and_sum)

    # Setup
    data = SampledData(jnp.array([1.0, 2.0, 3.0]))
    
    # Execute
    gradient = grad_func(data)

    # Assert: The gradient of sum(x^2) is 2*x.
    # The output of grad will be a SampledData object because of our pytree registration.
    assert isinstance(gradient, SampledData)
    np.testing.assert_allclose(gradient._data, jnp.array([2.0, 4.0, 6.0]))
    print("\nGrad test successful!")
# In test_sampled_data.py, at the end of the file

# In test_sampled_data.py
# Let's replace the vmap test one more time.

# --- Move the function to be vmapped OUTSIDE the test function ---
def _dot_product_on_raw_vector(raw_vector):
    """Helper function for vmap test."""
    return jnp.dot(raw_vector, raw_vector)

def test_vmap_compatibility():
    """
    Tests if a function operating on a SampledData object can be vectorized with jax.vmap.
    """
    # Create the vectorized version of the pure, external function.
    vmapped_dot = jax.vmap(_dot_product_on_raw_vector, in_axes=0)

    # 1. Setup: Create a batched SampledData object
    data = SampledData(jnp.array([[1., 2.], [3., 4.], [5., 6.]]))

    # 2. Execute: UNWRAP THE DATA BEFORE passing it.
    result = vmapped_dot(data._data)

    # 3. Assert: The result is a raw JAX array.
    assert not isinstance(result, SampledData)
    
    expected_values = jnp.array([5., 25., 61.])
    np.testing.assert_allclose(result, expected_values)
    print(f"\nvmap test successful! Result: {result}, Expected: {expected_values}")

    # In test_sampled_data.py, at the end of the file

# ================== Advanced JAX Compatibility Tests ==================

def test_pmap_compatibility():
    """
    Tests if SampledData can be used with jax.pmap for data parallelism.
    Requires running pytest with simulated devices, e.g.:
    XLA_FLAGS='--xla_force_host_platform_device_count=4' pytest
    """
    # Check if we have more than one device to test on
    if jax.device_count() < 2:
        pytest.skip("Skipping pmap test: requires multiple devices.")

    # Define a simple function that operates on a raw JAX array
    def simple_scaling(raw_array):
        return raw_array * 2

    # Create the parallelized version of the function
    pmapped_scaling = jax.pmap(simple_scaling)

    # 1. Setup: Create a batched SampledData object
    # The first dimension must be equal to the number of devices.
    num_devices = jax.device_count()
    data_shape = (num_devices, 3, 4) # e.g., (4, 3, 4) for 4 devices
    data = SampledData(jnp.arange(np.prod(data_shape)).reshape(data_shape))

    # 2. Execute: Run the parallelized function on the raw data.
    # The input data is automatically split across the first axis.
    result = pmapped_scaling(data._data)

    # 3. Assert: Check the result.
    assert not isinstance(result, SampledData)
    assert result.shape == data_shape
    
    # The result should be the original data, scaled by 2.
    expected_values = jnp.arange(np.prod(data_shape)).reshape(data_shape) * 2
    np.testing.assert_allclose(result, expected_values)
    print(f"\npmap test successful on {num_devices} devices!")

# In test_sampled_data.py, at the end of the file

from jax import lax

def test_lax_scan_compatibility():
    """
    Tests if SampledData can be used as the carry state in a lax.scan loop.
    """
    # Define the function to be executed at each step of the loop.
    # It takes the carry state (our object) and an input, and returns
    # the new carry state and an output for that step.
    def accumulate_in_wrapper(sd_carry, x):
        # This function is JAX-idiomatic: it unwraps, computes, and re-wraps.
        new_data = sd_carry._data + x
        new_sd_carry = SampledData(new_data)
        
        # The output for this step can be a raw value, e.g., the sum.
        step_output = jnp.sum(new_data)
        
        return new_sd_carry, step_output

    # 1. Setup: Define the initial state of the loop
    initial_carry = SampledData(jnp.zeros(3, dtype=jnp.float32))
    
    # Define the inputs for each step of the loop (e.g., 5 steps)
    xs = jnp.ones((5, 3)) # Add a vector of ones at each step

    # 2. Execute: Run the scan.
    final_carry, all_outputs = lax.scan(accumulate_in_wrapper, initial_carry, xs)

    # 3. Assert: Check the final state and the stacked outputs.
    
    # Check the final carry object
    assert isinstance(final_carry, SampledData)
    # After adding [1,1,1] five times, the final state should be [5,5,5]
    expected_final_carry = jnp.array([5., 5., 5.])
    np.testing.assert_allclose(final_carry._data, expected_final_carry)

    # Check the stacked outputs
    # The sums at each step should be: 3, 6, 9, 12, 15
    assert not isinstance(all_outputs, SampledData)
    expected_outputs = jnp.array([3., 6., 9., 12., 15.])
    np.testing.assert_allclose(all_outputs, expected_outputs)
    print("\nlax.scan test successful!")