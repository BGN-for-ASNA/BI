from Darray import*
from functools import partial
@partial(jit, static_argnums=(1, 2,))
def vec_to_mat(arr, N, K):
    return jnp.reshape(arr, (N, K))

@jit
def jax_LinearOperatorDiag(s, cov):    
    def multiply_with_s(a):
        return jnp.multiply(a, s)
    vectorized_multiply = vmap(multiply_with_s)
    return jnp.transpose(vectorized_multiply(cov))

import jax.numpy as jnp

@jit
def diag_pre_multiply(v, m):
    return jnp.matmul(jnp.diag(v), m)

@jit
def random_centered(sigma, cor_mat, offset_mat):
    """Generate the centered matrix of random factors 

    Args:
        sigma (vector): Prior, vector of length N
        cor_mat (2D array): correlation matrix, cholesky_factor_corr of dim N, N
        offset_mat (2D array): matrix of offsets, matrix of dim N*k

    Returns:
        _type_: 2D array
    """
    return jnp.dot(diag_pre_multiply(sigma, cor_mat), offset_mat)

softmax_fn = jax.vmap(jax.nn.softmax, in_axes=(0,))

@jit
def random_centered2(sigma, cor_mat, offset_mat):
    return ((sigma[..., None] * cor_mat) @ offset_mat)

@jit
def cov_GPL2(x, sq_eta, sq_rho, sq_sigma):
    N = x.shape[0]
    K = sq_eta * jnp.exp(-sq_rho * jnp.square(x))
    K = K.at[jnp.diag_indices(N)].add(sq_sigma)
    return K
