import jax
import jax.numpy as jnp
from jax import jit
from BI.Distributions.np_dists import UnifiedDist as dist
dist = dist()

# Gaussian process related functions ----------------------------------------
@jit
def cov_GPL2(x, sq_eta, sq_rho, sq_sigma):
    """
    Computes the covariance matrix for a Gaussian Process using the 
    squared exponential kernel, version L2 (squared Euclidean distance).

    Args:
        x: Distance matrix between points.
        sq_eta: Squared variance parameter (eta^2).
        sq_rho: Squared length scale parameter (rho^2).
        sq_sigma: Squared noise variance parameter (sigma^2).

    Returns:
        K: Covariance matrix incorporating the squared exponential kernel, version L2.
    """
    N = x.shape[0]
    K = sq_eta * jnp.exp(-sq_rho * jnp.square(x))
    K = K.at[jnp.diag_indices(N)].add(sq_sigma)
    return K

class gaussian:
    """Class for handling Gaussian process related operations in JAX, including distance matrix computation and kernel functions (squared exponential, periodic, periodic local)."""
    def __init__(self) -> None:
        pass
    
    @staticmethod
    @jit
    def distance_matrix(array):
        """Compute the distance matrix.
        Args:
            array (array): Input array representing the data points.
        Returns:
            array: The distance matrix computed using the absolute differences between data points.
        """
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
    def gaussian_process(Dmat, etasq = None, rhosq = None, sigmaq = 0.01):
        """Gaussian Process Model with Cholesky Decomposition L2
        Args:
            Dmat (array): Input array representing the distance matrix.
            etasq (float): Length-scale parameter of the squared exponential kernel.
            rhosq (float): Length-scale parameter of the squared exponential kernel.
            sigmaq (float): Scale parameter of the squared exponential kernel.
        Returns:
            array: The covariance matrix computed using the squared exponential kernel.
        """
        if etasq is None:
            etasq = dist.exponential(2, name = 'etasq')
        if rhosq is None:
            rhosq = dist.exponential(0.5, name = 'rhosq')

        SIGMA = cov_GPL2(Dmat, etasq, rhosq, sigmaq)
        return  dist.multivariate_normal(0, SIGMA, name='kernel')

    """Class for handling Gaussian process related operations in JAX, including distance matrix computation and kernel functions (squared exponential, periodic, periodic local)."""
    def __init__(self) -> None:
        pass

    @staticmethod
    @jit
    def distance_matrix(array):
        """Compute the distance matrix.
        Args:
            array (array): Input array representing the data points.
        Returns:
            array: The distance matrix computed using the absolute differences between data points.
        """
        return jnp.abs(array[:, None] - array[None, :])

    @staticmethod
    @jit
    def kernel_sq_exp(m, z, sq_alpha=0.5, sq_rho=0.1, delta=0):
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
    def kernel_periodic(m, sigma=1.0, length_scale=1.0, period=1.0, **kwargs):
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
        return sigma**2 * jnp.exp(-2*jnp.sin(jnp.pi * m / period)**2 / length_scale**2)

    @staticmethod
    @jit
    def kernel_periodic_local(m, sigma=1.0, length_scale=1.0, period=1.0, **kwargs):
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
        # Ensure the correct squared exponential term is used
        sq_exp_term = jnp.exp(-m**2 / (2 * length_scale**2))
        periodic_term = sigma**2 * jnp.exp(-2 * jnp.sin(jnp.pi * m / period)**2 / length_scale**2)
        return periodic_term * sq_exp_term

    @staticmethod
    def gaussian_process2(Dmat, kernel_type="sq_exp", **kernel_params):
        """Gaussian Process Model with selectable kernel.

        Args:
            Dmat (array): Input array representing the distance matrix.
            kernel_type (str): The type of kernel to use. Options are:
                               'sq_exp', 'periodic', 'periodic_local'.
            **kernel_params: Keyword arguments for the selected kernel function.
                               For 'sq_exp': z, sq_alpha, sq_rho, delta
                               For 'periodic': sigma, length_scale, period
                               For 'periodic_local': sigma, length_scale, period

        Returns:
            For 'sq_exp': A tuple (K, L_SIGMA, k).
            For other kernels: The covariance matrix SIGMA.
            In a probabilistic programming context, this would typically return a distribution.
        """
        if kernel_type == "sq_exp":
            # Set default values for Pyro/NumPyro distributions if not provided
            z = kernel_params.get('z')
            if z is None:
                raise ValueError("Parameter 'z' must be provided for the squared exponential kernel.")

            sq_alpha = kernel_params.get('sq_alpha')
            if sq_alpha is None:
                sq_alpha = dist.Exponential(1.0) # Example prior

            sq_rho = kernel_params.get('sq_rho')
            if sq_rho is None:
                sq_rho = dist.Exponential(0.5) # Example prior

            delta = kernel_params.get('delta', 1e-6)

            return Mgaussian.kernel_sq_exp(Dmat, z, sq_alpha, sq_rho, delta)

        elif kernel_type == "periodic":
            SIGMA = Mgaussian.kernel_periodic(Dmat, **kernel_params)
            # In a modeling context, you might do something like this:
            # return dist.MultivariateNormal(loc=jnp.zeros(Dmat.shape[0]), covariance_matrix=SIGMA)
            return SIGMA

        elif kernel_type == "periodic_local":
            SIGMA = Mgaussian.kernel_periodic_local(Dmat, **kernel_params)
            # return dist.MultivariateNormal(loc=jnp.zeros(Dmat.shape[0]), covariance_matrix=SIGMA)
            return SIGMA
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}. Available options are 'sq_exp', 'periodic', 'periodic_local'.")