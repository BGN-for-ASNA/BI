import inspect
from numpyro import sample as lk
from numpyro import deterministic
from utils.unified_dists import UnifiedDist as dist
from Mutils import Mgaussian
from Mutils import factors
import jax 
from jax import jit
import jax.numpy as jnp

gaussian = Mgaussian()
factor = factors()

from jax import vmap
#' Test
#region
#from Darray import *
from functools import partial
import jax as jax
import jax.numpy as jnp
from jax import jit

# vector related functions -----------------------------------
@partial(jit, static_argnums=(1, 2,))
def vec_to_mat_jax(arr, N, K):
    return jnp.tile(arr, (N, K))

# Matrices related functions ------------------------------------------------------------------
def upper_tri(array, diag=1):
    """Extracts the upper triangle elements of a 2D JAX array.

    Args:
        array (2D array): A JAX 2D array.
        diag (int): Integer indicating if diagonal must be kept or not.
                    diag=1 excludes the diagonal, diag=0 includes it.
    """
    upper_triangle_indices = jnp.triu_indices(array.shape[0], k=diag)
    upper_triangle_elements = array[upper_triangle_indices]
    return upper_triangle_elements
# JIT compile the function with static_argnums
get_upper_tri = jit(upper_tri, static_argnums=(1,))


def lower_tri(array, diag=-1):
    """Extracts the lower triangle elements of a 2D JAX array.

    Args:
        array (2D array): A JAX 2D array.
        diag (int): Integer indicating if diagonal must be kept or not.
                    diag=0 includes the diagonal, diag=-1 excludes it.
    """
    lower_triangle_indices = jnp.tril_indices(array.shape[0], k=diag)
    lower_triangle_elements = array[lower_triangle_indices]
    return lower_triangle_elements
# JIT compile the function with static_argnums
get_lower_tri = jit(lower_tri, static_argnums=(1,))

def get_tri(array, type='upper', diag=0):
    """Extracts the upper, lower, or both triangle elements of a 2D JAX array.

    Args:
        array (2D array): A JAX 2D array.
        type (str): A string indicating which part of the triangle to extract.
                    It can be 'upper', 'lower', or 'both'.
        diag (int): Integer indicating if diagonal must be kept or not.
                    diag=1 excludes the diagonal, diag=0 includes it.

    Returns:
        If argument type is 'upper', 'lower', it return a 1D JAX array containing the requested triangle elements.
        If argument type is 'both', it return a 2D JAX array containing the the first column the lower triangle and in the second ecolumn the upper triangle
    """
    if type == 'upper':
        upper_triangle_indices = jnp.triu_indices(array.shape[0], k=diag)
        triangle_elements = array[upper_triangle_indices]
    elif type == 'lower':
        lower_triangle_indices = jnp.tril_indices(array.shape[0], k=-diag)
        triangle_elements = array[lower_triangle_indices]
    elif type == 'both':
        upper_triangle_indices = jnp.triu_indices(array.shape[0], k=diag)
        lower_triangle_indices = jnp.tril_indices(array.shape[0], k=-diag)
        upper_triangle_elements = array[upper_triangle_indices]
        lower_triangle_elements = array[lower_triangle_indices]
        triangle_elements = jnp.stack((upper_triangle_elements,lower_triangle_elements), axis = 1)
    else:
        raise ValueError("type must be 'upper', 'lower', or 'both'")

    return triangle_elements

    
@jit
def mat_to_edgl_jax(mat):
    N = mat.shape[0]
    # From to 
    urows, ucols   = jnp.triu_indices(N, k=1)
    ft = mat[(urows,ucols)]
    m2 = jnp.transpose(mat)
    tf = m2[(urows,ucols)]
    return jnp.stack([tf, ft], axis = -1)


class manip():
    def __init__(self) -> None:
        pass

    @staticmethod 
    @jit
    def logit(x):
        return jnp.log(x / (1 - x))

    # Matrix manipulations -------------------------------------
    @staticmethod 
    @partial(jit, static_argnums=(1, ))
    def vec_to_mat(vec, shape = ()):
        return jnp.tile(vec, shape)

    def get_tri(self, array, type='upper', diag=0):
        return get_tri(array, type=type, diag=diag)
    
    @staticmethod 
    @jit
    def mat_to_edgl(mat):
        N = mat.shape[0]
        # From to 
        urows, ucols   = jnp.triu_indices(N, k=1)
        ft = mat[(urows,ucols)]

        m2 = jnp.transpose(mat)
        tf = m2[(urows,ucols)]
        return jnp.stack([ft, tf], axis = -1)

    @staticmethod 
    @partial(jit, static_argnums=(1, ))
    def edgl_to_mat(edgl, N_id):
        m = jnp.zeros((N_id,N_id))
        urows, ucols   = jnp.triu_indices(N_id, 1)
        m = m.at[(ucols, urows)].set(edgl[:,1])
        m = m.at[(urows, ucols)].set(edgl[:,0])
        return m
    
    @staticmethod 
    @jit
    def remove_diagonal(arr):
        n = arr.shape[0]
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("Array must be square to remove the diagonal.")

        # Create a mask for non-diagonal elements
        mask = ~jnp.eye(n, dtype=bool)

        # Apply the mask to the array to get non-diagonal elements
        non_diag_elements = arr[mask]  # Reshape as needed, here to an example shape
    
        return non_diag_elements
    
    @staticmethod 
    @jit    
    def vec_node_to_edgle(sr):
        """_summary_

        Args:
            sr (2D array): Each column represent an characteristic or effect and  each row represent the value of i for the characteristic of the given column

        Returns:
            (2D array): return and edgelist of all dyads combination (excluding diagonal).
            First column represent the value fo individual i  in the first column of argument sr, the second column the value of j in the second column of argument sr
        """
        N = sr.shape[0]
        lrows, lcols = jnp.tril_indices(N, k=-1)
        urows, ucols = jnp.triu_indices(N, k=1)
        ft = sr[urows,0]
        tf = sr[ucols,1]
        return jnp.stack([ft, tf], axis = -1)
    
