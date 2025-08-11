
from BI.Utils.np_dists import UnifiedDist as dist
import jax.numpy as jnp
def pca(X, latent_dim): 
    """
    Implements a PCA model using JAX and UnifiedDist.
    Args:
        X (jnp.ndarray): Training data.
        latent_dim (int): Dimensionality of the latent space.
    Returns:
        None    
    """
    num_datapoints = X.shape[0]
    data_dim = X.shape[1]

    epsilon = dist.exponential(1, name='epsilon')
    w = dist.normal(0, 1, shape=(data_dim, latent_dim), name='w')
    z = dist.normal(0, 1, shape=(latent_dim, num_datapoints), name='z')
    tmp = w @ z
    dist.normal(tmp.T, epsilon, obs = X)  
    
def align_pca(real_data,posteriors):
    """Aligns the PCA components of the real data with the estimated components from the posterior samples.
    Args:
        real_data (jnp.ndarray): The true PCA components from the real data.
        posteriors (jnp.ndarray): The estimated PCA components from the posterior samples.
    Returns:
        jnp.ndarray: The aligned estimated PCA components.
    """
    dot_product = jnp.dot(real_data, posteriors)
    # Align signs if necessary
    if dot_product < 0:
        posteriors = -posteriors
    return posteriors

