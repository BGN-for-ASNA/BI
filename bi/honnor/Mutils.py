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



@jit
def mat_distance(array):
    return jnp.abs(array[:, None] - array[None, :])


@jit
def sq_exp_kernel(m, sq_alpha=0.5, sq_rho=0.1, delta=0):
    """Squared Exponential Kernel.

    The SE kernel is a widely used kernel in Gaussian processes (GPs) and support vector machines (SVMs). It has some desirable properties, such as universality and infinite differentiability. This function computes the covariance matrix using the squared exponential kernel.

    Args:
        m (array): Input array representing the absolute distances between data points.
        sq_alpha (float, optional): Scale parameter of the squared exponential kernel. Defaults to 0.5.
        sq_rho (float, optional): Length-scale parameter of the squared exponential kernel. Defaults to 0.1.
        delta (int, optional): Delta value to be added to the diagonal of the covariance matrix. Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - K (array): The covariance matrix computed using the squared exponential kernel.
            - cov (array): A masked covariance matrix with the upper triangular part set to zero.
    """
    # Get the number of data points
    N = m.shape[0]
    
    # Compute the kernel matrix using the squared exponential kernel
    K = sq_alpha * jnp.exp(-sq_rho *  jnp.square(m))
    
    # Set the diagonal elements of the kernel matrix
    K = K.at[jnp.diag_indices(N)].set(sq_alpha + delta)
    
    # Create a mask for the upper triangular part of the covariance matrix
    mask = jnp.triu(jnp.ones_like(K, dtype=bool))
    
    # Apply the mask to set the upper triangular part of the covariance matrix to zero
    cov = jnp.where(mask, K, 0)
    
    return K, cov

@jit
def periodic_kernel(m, sigma=1, length_scale=1.0, period=1.0):
    """Periodic Kernel.

    The periodic kernel is often used in Gaussian processes (GPs) for modeling functions with periodic behavior.

    Args:
        m (array): Input array representing the absolute distances between data points.
        sigma (float, optional): Scale parameter of the kernel. Defaults to 1.0.
        length_scale (float, optional): Length scale parameter of the kernel. Defaults to 1.0.
        period (float, optional): Period parameter of the kernel. Defaults to 1.0.

    Returns:
        array: The covariance matrix computed using the periodic kernel.
    """    
    # Compute the kernel matrix using the squared exponential kernel
    return sigma**2 * jnp.exp(-2*jnp.sin(jnp.pi * m / period)**2 / length_scale**2) 

@jit
def local_periodic_kernel(m, sigma=1, length_scale=1.0, period=1.0):
    """Locally Periodic Kernel

    A SE kernel times a periodic results in functions which are periodic, but which can slowly vary over time.

    Args:
        m (array): Input array representing the absolute distances between data points.
        sigma (float, optional): Scale parameter of the kernel. Defaults to 1.0.
        length_scale (float, optional): Length scale parameter of the kernel. Defaults to 1.0.
        period (float, optional): Period parameter of the kernel. Defaults to 1.0.

    Returns:
        array: The covariance matrix computed using the periodic kernel.
    """    
    # Compute the kernel matrix using the squared exponential kernel
    return sigma**2 * jnp.exp(-2*jnp.sin(jnp.pi * m / period)**2 / length_scale**2)  * jnp.exp(-(m**2/ 2*length_scale**2))