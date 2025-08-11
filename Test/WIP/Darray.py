
import jax as jax
import jax.numpy as jnp
from jax import jit
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
from jax import random
import random as r
init_key, sample_key = random.split(random.PRNGKey(int(r.randint(0, 10000000))))
init_key = jnp.array(init_key)

# Distributed arrays and automatic parallelization : https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
def sharding(cores = None):
    """
    Create sharding configuration for distributed computation using JAX PositionalSharding.

    Returns:
        jax.experimental.pjit.PositionalSharding: Sharding configuration for distributed computation.
    """
    if cores == None:
        cores = jax.local_device_count(backend=None)
    sharding = PositionalSharding(mesh_utils.create_device_mesh(cores, devices = jax.devices()[0:cores]))
    return sharding
#sharding(cores = 32)

import jax.numpy as jnp
import itertools

def valid_reshape_combinations(shape):
    """
    Generates valid reshape combinations for a given shape.
    
    Args:
    - shape: Tuple of integers representing the shape of the array.
    
    Returns:
    - List of tuples representing valid reshape combinations.
    """
    factors = []
    for i in range(1, int(jnp.sqrt(shape)) + 1):
        if shape % i == 0:
            factors.append((i, shape // i))
    
    combinations = []
    for factor1, factor2 in factors:
        combinations.append((factor1, factor2))
        if factor1 != factor2:
            combinations.append((factor2, factor1))
    
    return combinations

# Example usage:
#combinations = valid_reshape_combinations(jax.local_device_count(backend=None))
#print(combinations)

def split_array_to_cores(x, cores = None):
    """
    Splits an array `x` across available CPU cores using JAX's PositionalSharding.

    Args:
        x (numpy.ndarray or jax.interpreters.xla.DeviceArray): Input array to be split.

    Returns:
        jax.interpreters.xla.DeviceArray: The input array split across available CPU cores.
    """
    if cores is None:
        n = jax.local_device_count(backend=None)    
        shapes = jax.numpy.array(x.shape)
        if any( i < n  for i in shapes):
            cores = jax.numpy.min(shapes)
        else:
            cores = jax.local_device_count(backend=None)        
    s = sharding(cores)
    valid_shapes = valid_reshape_combinations(cores)

    return jax.device_put(x, s.reshape(valid_shapes[len(valid_shapes)-1]))
