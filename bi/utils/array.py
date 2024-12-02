#from Darray import*
import jax as jax
import jax.numpy as jnp
from jax import jit
from jax import vmap
import numpyro
from utils.unified_dists import tfpLight as dist

# Random factors related functions --------------------------------------------
@jit
def jax_LinearOperatorDiag(s, cov):    
    def multiply_with_s(a):
        return jnp.multiply(a, s)
    vectorized_multiply = vmap(multiply_with_s)
    return jnp.transpose(vectorized_multiply(cov))

#@jit
#def diag_pre_multiply(v, m):
#    return jnp.matmul(jnp.diag(v), m)#

#@jit
#def random_centered_jax(sigma, cor_mat, offset_mat):
#    """Generate the centered matrix of random factors #

#    Args:
#        sigma (vector): Prior, vector of length N
#        cor_mat (2D array): correlation matrix, cholesky_factor_corr of dim N, N
#        offset_mat (2D array): matrix of offsets, matrix of dim N*k#

#    Returns:
#        _type_: 2D array
#    """
#    return jnp.dot(diag_pre_multiply(sigma, cor_mat), offset_mat)

#@jit
#def random_centered2(sigma, cor_mat, offset_mat):
#    return ((sigma[..., None] * cor_mat) @ offset_mat)

class factors:
    def __init__(self) -> None:
        pass

    @staticmethod 
    @jit 
    def diag_pre_multiply(v, m):
        return jnp.matmul(jnp.diag(v), m)

    @staticmethod 
    @jit    
    def random_centered(sigma, cor_mat, offset_mat):
        """Generate the centered matrix of random factors 

        Args:☺
            sigma (vector): Prior, vector of length N
            cor_mat (2D array): correlation matrix, cholesky_factor_corr of dim N, N
            offset_mat (2D array): matrix of offsets, matrix of dim N*k

        Returns:
            _type_: 2D array
        """
        #return jnp.dot(factors.diag_pre_multiply(sigma, cor_mat), offset_mat).T
        return (factors.diag_pre_multiply(sigma, cor_mat) @ offset_mat).T

    
# Gaussian process related functions ----------------------------------------
@jit
def cov_GPL2(x, sq_eta, sq_rho, sq_sigma):
    N = x.shape[0]
    K = sq_eta * jnp.exp(-sq_rho * jnp.square(x))
    K = K.at[jnp.diag_indices(N)].add(sq_sigma)
    return K

@jit
def sq_exp_kernel(m, sq_alpha=0.5, sq_rho=0.1, delta=0, only_K = True):
    """Squared Exponential Kernel.

    The SE kernel is a widely used kernel in Gaussian processes (GPs) and support vector machines (SVMs). It has some desirable properties, such as universality and infinite differentiability. This function computes the covariance matrix using the squared exponential kernel.

    Args:
        m (array): Input array representing the absolute distances between data points.
        sq_alpha (float, optional): Scale parameter of the squared exponential kernel. Defaults to 0.5.
        sq_rho (float, optional): Length-scale parameter of the squared exponential kernel. Defaults to 0.1.
        delta (int, optional): Delta value to be added to the diagonal of the covariance matrix. Defaults to 0.
        only_K (bool, , optional): Return only the covariance matrix
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
    
    if only_K:
        return K
    
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

class Mgaussian:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    @jit
    def distance_matrix(array):
        return jnp.abs(array[:, None] - array[None, :])
    
    @staticmethod
    @jit
    def kernel_sq_exp(m,z, sq_alpha=0.5, sq_rho=0.1, delta=0):
        """Squared Exponential Kernel.

        The SE kernel is a widely used kernel in Gaussian processes (GPs) and support vector machines (SVMs). It has some desirable properties, such as universality and infinite differentiability. This function computes the covariance matrix using the squared exponential kernel.

        Args:
            m (array): Input array representing the absolute distances between data points.
            z (array): Input array representing the random effect.
            sq_alpha (float, optional): Scale parameter of the squared exponential kernel. Defaults to 0.5.
            sq_rho (float, optional): Length-scale parameter of the squared exponential kernel. Defaults to 0.1.
            delta (int, optional): Delta value to be added to the diagonal of the covariance matrix. Defaults to 0.

        Returns:
            tuple: A tuple containing:
                - K (array): The covariance matrix computed using the squared exponential kernel.
                - L_SIGMA (array): Cholesky decomposition of K.
                - k: Kernel function
        """
        # Get the number of data points
        N = m.shape[0]

        # Compute the kernel matrix using the squared exponential kernel
        K = sq_alpha * jnp.exp(-sq_rho *  jnp.square(m))

        # Set the diagonal elements of the kernel matrix
        K = K.at[jnp.diag_indices(N)].set(sq_alpha + delta)

        # Compute the Cholesky decomposition of the kernel matrix
        L_SIGMA = jnp.linalg.cholesky(K)

        # Compute the kernel function
        k = (L_SIGMA @ z[..., None])[..., 0]

        return K, L_SIGMA, k
        
    @staticmethod
    @jit
    def kernel_periodic(m, sigma=1, length_scale=1.0, period=1.0):
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

    @staticmethod
    @jit
    def kernel_periodic_local(m, sigma=1, length_scale=1.0, period=1.0):
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

    @staticmethod
    def gaussian_process(Dmat, etasq, rhosq, sigmaq):
        SIGMA = cov_GPL2(Dmat, etasq, rhosq, sigmaq)
        L_SIGMA = jnp.linalg.cholesky(SIGMA)
        z = dist.normal('z', 0, 1, sample_shape= [10])
        k = numpyro.deterministic("k", (L_SIGMA @ z[..., None])[..., 0])
        return k
