#from Darray import*
import jax as jax
import jax.numpy as jnp
from jax import jit
from jax import vmap
import numpyro
from BI.Utils.np_dists import UnifiedDist as dist

# Random factors related functions --------------------------------------------
@jit
def jax_LinearOperatorDiag(s, cov):    
    def multiply_with_s(a):
        return jnp.multiply(a, s)
    vectorized_multiply = vmap(multiply_with_s)
    return jnp.transpose(vectorized_multiply(cov))

class effects:
    """Class for handling specific factor operations in JAX, including diagonal matrix multiplication and random factor centered generation."""
    def __init__(self) -> None:
        pass

    @staticmethod 
    @jit 
    def diag_pre_multiply(v, m):
        """Dot product of a diagonal matrix with a matrix.

        Args:
            v (1D jax array): Vector of diagonal elements.
            m (2D jax array): Matrix to be multiplied.

        Returns:
            2D jax array: Dot product of a diagonal matrix with a matrix.
        """
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
        return (effects.diag_pre_multiply(sigma, cor_mat) @ offset_mat).T


    @staticmethod 
    def varying_intercept_slope(
        N_group, 
        group,
        global_intercept, 
        global_slope,
        group_name = 'group',
        group_std =  None,
        L_corr = None,
        sample=False,
    ):
        """
        Models a varying intercept and varying slope for a continuous predictor.

        This uses a "centered" parameterization to sample a correlated
        [intercept, slope] pair for each group from a Multivariate Normal distribution.

        Args:
            N_groups (int): The total number of unique groups (e.g., 20 schools).
            group_idx (array): An array of integer indices for the group of each observation.
            global_intercept_mean (float or distribution): The hyperprior for the mean of the intercepts.
            global_slope_mean (float or distribution): The hyperprior for the mean of the slopes.
            group_name (str, optional): A descriptive name for the grouping factor.
            group_std (distribution, optional): Prior for the standard deviations of the
                intercept and slope. Must have shape (2,). Defaults to Exponential(1).
            L_corr (distribution, optional): Prior for the Cholesky factor of the 2x2
                correlation matrix between the intercept and slope. Defaults to LKJ(2, 2).

        Returns:
            tuple: A tuple containing two arrays:
                - varying_intercepts (array): The intercept for each observation.
                - varying_slopes (array): The slope for each observation.
        """

        # 1. Handle default hyperpriors.
        if group_std is None:
            group_std = dist.exponential(1, shape=(2,),  name = f'{group_name}_std', sample = sample)
            
        if L_corr is None:
            L_corr = dist.lkj(2, 2, name = f'L_corr_{group_name}', sample = sample)

        cov = jnp.outer(group_std, group_std) * L_corr

        group_params =  dist.multivariate_normal(
            jnp.stack([global_intercept, global_slope]), 
            cov, 
            shape = (N_group,), 
            name = f'{group_name}_params', 
            sample = sample
        )    
        varying_intercept, varying_slope= group_params[:, 0], group_params[:, 1]
        varying_intercept = group_params[:, 0][group]
        varying_slope = group_params[:, 1][group]
        return  varying_intercept, varying_slope
    
    
    @staticmethod
    def varying_slope(
        category_idx,
        grouping_idx,
        num_categories,
        num_groups,
        L_Rho=None,
        sigma=None,
        group_name='group',
        sample=False
    ):
        """Generates hierarchical varying effects using a non-centered parameterization.

        This is a general-purpose component for building hierarchical models. It models
        a set of related effects (e.g., slopes or intercepts) that vary by a grouping
        factor. The effects for each group are "pooled" together, meaning the model
        learns a shared distribution for them, allowing information to be shared
        among groups.

        This is useful for modeling:
        - The effect of different treatments ('categories') varying across subjects ('groups').
        - The performance of students ('groups') across different school subjects ('categories').
        - The changing approval rating ('category') of a policy over time in different countries ('groups').

        Args:
            category_idx (array): An array of integer indices for the category of the
                effect for each observation. For example, if you have 4 treatments,
                this would be an array of numbers from 0 to 3.
            grouping_idx (array): An array of integer indices for the grouping factor
                for each observation (e.g., subject ID, school ID, country ID).
            num_categories (int): The total number of unique categories for the effects
                (e.g., 4 treatments). This defines the dimensionality of the effects.
            num_groups (int): The total number of unique groups (e.g., 7 subjects).
            L_Rho (distribution, optional): The prior for the Cholesky factor of the
                correlation matrix between the effects. If None, defaults to a weakly
                informative LKJCholesky(num_categories, 2) prior.
            sigma (distribution, optional): The prior for the standard deviations (scales)
                of the effects for each category. If None, defaults to a weakly
                informative Exponential(1) prior.
            group_name (str, optional): A descriptive name for the grouping factor, used
                to create unique parameter names in the model trace. Defaults to 'group'.

        Returns:
            array: An array containing the specific varying effect for each
                   corresponding observation in the input arrays.
        """

        # 1. Define default hyperpriors if not provided by the user.
        # These priors define the multivariate distribution from which the
        # individual group effects are drawn.
        if L_Rho is None:
            # The LKJ Cholesky prior is a distribution over correlation matrices.
            # An eta > 1 (here, 2) encourages weaker correlations.
            L_Rho = m.dist.lkj_cholesky(num_categories, 2, name=f"L_Rho_{group_name}", sample=sample)
        if sigma is None:
            # The Exponential prior ensures the standard deviations are positive.
            sigma = m.dist.exponential(1, name=f'sigma_{group_name}', shape=(num_categories,), sample=sample)

        # 2. Implement the non-centered parameterization for efficiency.
        # We sample uncorrelated standard normal values and then transform them.
        # This improves the geometry of the posterior for the sampler. [1, 2, 3, 4, 5]
        z = m.dist.normal(0, 1, name=f"z_{group_name}", shape=(num_categories, num_groups), sample=sample)

        # 3. Transform the standard normal samples into correlated varying effects.
        # This operation combines the scale (sigma), correlation (L_Rho), and the
        # base samples (z). The result 'effects' is a matrix of shape
        # (num_groups, num_categories).
        effects = ((sigma[..., None] * L_Rho) @ z).T

        # 4. Return the specific effect for the category and group of each observation.
        # This uses advanced indexing to pick the correct value from the 'effects' matrix
        # for each row of data.
        return effects[grouping_idx, category_idx]

    @staticmethod
    def varying_intercept(
        N_groups,
        group_idx,
        a_bar=None,
        sigma=None,
        group_name='group',
        sample=False
    ):
        """
        Varying intercepts model using a non-centered parameterization.

        Args:
            N_groups (int): The total number of unique groups.
            group_idx (array): An array of integer indices specifying the group for each observation.
            a_bar (distribution, optional): The hyperprior for the mean of the group-level intercepts.
            sigma (distribution, optional): The hyperprior for the standard deviation of the group-level intercepts.

        Returns:
            array: An array of the appropriate group-level intercepts for each observation.
        """
        # 1. Handle default hyperpriors.
        if a_bar is None:
            a_bar = dist.normal(0., 1.5, name=f'global_intercept_{group_name}',sample=sample)
        if sigma is None:
            sigma = dist.exponential(1., name=f'sd_{group_name}',sample=sample)

        # 2. Define the distribution for the group-level intercepts ('alpha')
        # There is one alpha for each unique group.
        alpha = dist.normal(a_bar, sigma, shape=(N_groups,), name=f'intercept_{group_name}',sample=sample)

        # 3. Return the specific intercept corresponding to the group of each observation
        return alpha[group_idx]

    @staticmethod
    def varying_intercept_slope_non_centered(
        N_groups,
        group_idx,
        global_intercept_mean,
        global_slope_mean,
        group_name='group',
        sigma=None,
        L_Rho=None,
        sample=False,
    ):
        """
        Models a varying intercept and varying slope using a NON-CENTERED parameterization.

        Args:
            N_groups (int): The total number of unique groups.
            group_idx (array): An array of integer indices for the group of each observation.
            global_intercept_mean (float or distribution): The hyperprior for the mean of the intercepts.
            global_slope_mean (float or distribution): The hyperprior for the mean of the slopes.
            group_name (str, optional): A descriptive name for the grouping factor.
            sigma (distribution, optional): Prior for the standard deviations of the
                intercept and slope. Must have shape (2,). Defaults to Exponential(1).
            L_Rho (distribution, optional): Prior for the Cholesky factor of the 2x2
                correlation matrix. Defaults to LKJCholesky(2, 2).

        Returns:
            tuple: A tuple containing two arrays:
                - varying_intercepts (array): The intercept for each observation.
                - varying_slopes (array): The slope for each observation.
        """
        # 1. Handle default hyperpriors for scale (sigma) and correlation (L_Rho).
        if sigma is None:
            sigma = dist.exponential(1, shape=(2,), name=f'{group_name}_sigma', sample=sample)
        if L_Rho is None:
            L_Rho = dist.lkj_cholesky(2, 2, name=f'L_Rho_{group_name}', sample=sample)

        # 2. Sample the uncorrelated, standard normal "offsets".
        z = dist.normal(0, 1, name=f"z_{group_name}", shape=(2, N_groups), sample=sample)

        # 3. Transform the offsets into correlated deviations from zero.
        effects = ((sigma[..., None] * L_Rho) @ z).T

        # 4. Construct the final parameters by adding the global means.
        mean_vector = jnp.stack([global_intercept_mean, global_slope_mean])
        params_per_group = mean_vector + effects

        # 5. Select the correct parameter pair for each observation.
        params_for_obs = params_per_group[group_idx]

        # 6. Unpack and return the final intercepts and slopes.
        varying_intercepts = params_for_obs[:, 0]
        varying_slopes = params_for_obs[:, 1]
        return varying_intercepts, varying_slopes


    @staticmethod
    def varying_slope_non_centered(
        category_idx,
        grouping_idx,
        num_categories,
        num_groups,
        L_Rho=None,
        sigma=None,
        group_name='group',
        sample=False
    ):
        """Generates hierarchical varying effects for CATEGORIES using a non-centered parameterization."""

        # 1. Define default hyperpriors if not provided by the user.
        if L_Rho is None:
            L_Rho = m.dist.lkj_cholesky(num_categories, 2, name=f"L_Rho_{group_name}", sample=sample)
        if sigma is None:
            sigma = m.dist.exponential(1, name=f'sigma_{group_name}', shape=(num_categories,), sample=sample)

        # 2. Sample uncorrelated standard normal values ("offsets").
        z = m.dist.normal(0, 1, name=f"z_{group_name}", shape=(num_categories, num_groups), sample=sample)

        # 3. Transform the offsets into correlated varying effects.
        effects = ((sigma[..., None] * L_Rho) @ z).T

        # 4. Return the specific effect for the category and group of each observation.
        return effects[grouping_idx, category_idx]


    @staticmethod
    def varying_intercept_non_centered(
        N_groups,
        group_idx,
        a_bar=None,
        sigma=None,
        group_name='group',
        sample=False
    ):
        """
        Varying intercepts model using a NON-CENTERED parameterization.

        Args:
            N_groups (int): The total number of unique groups.
            group_idx (array): An array of integer indices specifying the group for each observation.
            a_bar (distribution, optional): The hyperprior for the mean of the group-level intercepts.
            sigma (distribution, optional): The hyperprior for the standard deviation of the group-level intercepts.

        Returns:
            array: An array of the appropriate group-level intercepts for each observation.
        """
        # 1. Handle default hyperpriors for the mean (a_bar) and scale (sigma).
        if a_bar is None:
            a_bar = dist.normal(0., 1.5, name=f'global_intercept_{group_name}', sample=sample)
        if sigma is None:
            sigma = dist.exponential(1., name=f'sd_{group_name}', sample=sample)

        # 2. Sample the standard normal "offsets", one for each unique group.
        z = dist.normal(0., 1., name=f'z_intercept_{group_name}', shape=(N_groups,), sample=sample)

        # 3. Construct the intercepts by scaling and shifting the offsets.
        alpha = a_bar + z * sigma

        # 4. Return the specific intercept corresponding to the group of each observation.
        return alpha[group_idx]
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

class Mgaussian:
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
    def gaussian_process(Dmat, etasq, rhosq, sigmaq):
        """Gaussian Process Model with Cholesky Decomposition L2
        Args:
            Dmat (array): Input array representing the distance matrix.
            etasq (float): Length-scale parameter of the squared exponential kernel.
            rhosq (float): Length-scale parameter of the squared exponential kernel.
            sigmaq (float): Scale parameter of the squared exponential kernel.
        Returns:
            array: The covariance matrix computed using the squared exponential kernel.
        """
        SIGMA = cov_GPL2(Dmat, etasq, rhosq, sigmaq)
        #L_SIGMA = jnp.linalg.cholesky(SIGMA)
        #z = dist.normal(0, 1, shape= shape, name = 'z')
        #k = numpyro.deterministic("k", (L_SIGMA @ z[..., None])[..., 0])
        return  numpyro.sample("k", dist.MultivariateNormal(0, SIGMA))
