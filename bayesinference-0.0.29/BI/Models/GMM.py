import jax
import jax.numpy as jnp
from jax import random, vmap
from BI.Utils.np_dists import UnifiedDist as dist
import numpyro.distributions as Dist
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import multivariate_normal
import seaborn as sns # Ensure seaborn is imported for palettes
import matplotlib.pyplot as plt

def gmm(data, K, initial_means): 
    """
    Gaussian Mixture Model with a fixed number of clusters K.
    Parameters:
    - data: Input data points (shape: [N, D] where N is the number of samples and D is the number of features).
    - K: The exact number of clusters.
    - initial_means: Initial means for the clusters (shape: [K, D]). If not provided, it is initialized using K-means.
    Returns:
    - A model that defines the GMM with K clusters.
    This model assumes that the data is generated from a mixture of K Gaussian distributions.
    The model estimates the means, covariances, and mixture weights for each cluster.
    The number of clusters K is fixed and must be specified in advance.
    """
    D = data.shape[1]  # Number of features
    alpha_prior = 0.5 * jnp.ones(K)
    w = dist.dirichlet(concentration=alpha_prior, name='weights') 

    with dist.plate("components", K): # Use fixed K
        mu = dist.multivariatenormal(loc=initial_means, covariance_matrix=0.1*jnp.eye(D), name='mu')        
        sigma = dist.halfcauchy(1, shape=(D,), event=1, name='sigma')
        Lcorr = dist.lkjcholesky(dimension=D, concentration=1.0, name='Lcorr')

        scale_tril = sigma[..., None] * Lcorr

    ## 3) marginal mixture over obs (this part remains almost identical)
    #with numpyro.plate('data', len(data)):
    #    assignment = numpyro.sample('assignment', dist.Categorical(w),infer={"enumerate": "parallel"}) 
    #    numpyro.sample('obs', dist.MultivariateNormal(mu[assignment,:][1], sigma[assignment][1]*jnp.eye(D)), obs=data)
    #    
    dist.mixturesamefamily(
        mixing_distribution=dist.categorical(probs=w, create_obj=True),
        component_distribution=dist.multivariatenormal(loc=mu, scale_tril=scale_tril, create_obj=True),
        name="obs",
        obs=data
    )

def predict_gmm(data,sampler):
    """
    Predicts the GMM density contours based on posterior samples and final labels.
    
    Parameters:
    - data: The input data points.
    - w_samps: Posterior samples of the weights.
    - mu_samps: Posterior samples of the means.
    - sigma_samps: Posterior samples of the standard deviations.
    - Lcorr_samps: Posterior samples of the Cholesky factors for correlation.
    - final_labels: Cluster labels for each data point.
    
    Returns:
    - None (plots the GMM density contours).
    """
    # 1. Calculate posterior mean of all model parameters
    posterior_samples = sampler.get_samples()
    w_samps = posterior_samples['weights']
    mu_samps = posterior_samples['mu']
    Lcorr_samps = posterior_samples['Lcorr']
    sigma_samps = posterior_samples['sigma']

    post_mean_w = jnp.mean(w_samps, axis=0)
    post_mean_mu =jnp.mean(mu_samps, axis=0)
    post_mean_sigma = jnp.mean(sigma_samps, axis=0)
    post_mean_Lcorr = jnp.mean(Lcorr_samps, axis=0)

    # Reconstruct the full covariance matrices
    post_mean_scale_tril = post_mean_sigma[..., None] * post_mean_Lcorr
    post_mean_cov = post_mean_scale_tril @ jnp.transpose(post_mean_scale_tril, (0, 2, 1))

    # ... (The entire co-clustering block to get final_labels) ...
    def get_cluster_probs(data, w, mu, sigma, Lcorr):
        scale_tril = sigma[..., None] * Lcorr
        log_liks = Dist.MultivariateNormal(mu, scale_tril=scale_tril).log_prob(data[:, None, :])
        log_probs = jnp.log(w) + log_liks
        norm_probs = jnp.exp(log_probs - jax.scipy.special.logsumexp(log_probs, axis=-1, keepdims=True))
        return norm_probs
    cluster_probs = jax.vmap(get_cluster_probs, in_axes=(None, 0, 0, 0, 0))(
        data, w_samps, mu_samps, sigma_samps, Lcorr_samps
    )
    similarity_matrix = (cluster_probs @ cluster_probs.transpose(0, 2, 1)).mean(axis=0)
    similarity_matrix_np = similarity_matrix
    distance_matrix = 1 - similarity_matrix_np
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    distance_matrix = distance_matrix.at[jnp.diag_indices_from(distance_matrix)].set(0.0)  # Set diagonal to 0
    distance_matrix = jnp.clip(distance_matrix, a_min=0.0, a_max=None)
    condensed_dist = squareform(distance_matrix)
    Z = linkage(condensed_dist, 'average')
    distance_threshold = 0.5 
    final_labels = fcluster(Z, t=distance_threshold, criterion='distance')

    return post_mean_w, post_mean_mu, post_mean_cov, final_labels

def plot_gmm(data,sampler):
    post_mean_w, post_mean_mu, post_mean_cov, final_labels = predict_gmm(data,sampler)
    # 2. Set up a grid of points to evaluate the GMM density
    x_min, x_max = data[:, 0].min() - 2, data[:, 0].max() + 2
    y_min, y_max = data[:, 1].min() - 2, data[:, 1].max() + 2
    xx, yy = jnp.meshgrid(jnp.linspace(x_min, x_max, 150),
                         jnp.linspace(y_min, y_max, 150))
    grid_points = jnp.c_[xx.ravel(), yy.ravel()]

    # 3. Calculate the PDF of the GMM on the grid
    num_components = post_mean_mu.shape[0]
    gmm_pdf = jnp.zeros(grid_points.shape[0])

    for k in range(num_components):
        # Get parameters for the k-th component
        weight = post_mean_w[k]
        mean = post_mean_mu[k]
        cov = post_mean_cov[k]

        # Calculate the PDF of this component and add its weighted value to the total
        component_pdf = multivariate_normal(mean=mean, cov=cov).pdf(grid_points)
        gmm_pdf += weight * component_pdf

    # Reshape the PDF values to match the grid shape
    Z = gmm_pdf.reshape(xx.shape)

    # 4. Create the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#f0f0f0') 
    ax.set_facecolor('#f0f0f0')

    # === FIX IS HERE ===
    # Dynamically create a color palette based on the number of clusters found
    unique_labels = jnp.unique(final_labels)
    n_clusters = len(unique_labels)
    # Using 'viridis' to match your first plot, but 'tab10' or 'Set2' are also good
    palette = sns.color_palette("viridis", n_colors=n_clusters) 

    # Create a mapping from each cluster label to its assigned color
    unique_labels = np.unique(final_labels)
    color_map = {label: palette[i] for i, label in enumerate(unique_labels)}
    # Create a list of colors for each data point corresponding to its cluster
    point_colors = [color_map[l] for l in final_labels]
    # === END OF FIX ===

    # Plot the data points using the dynamically generated colors
    ax.scatter(data[:, 0], data[:, 1], c=point_colors, s=15, alpha=0.9, edgecolor='white', linewidth=0.3)

    # Plot the density contours
    # Using a different colormap for the contours (e.g., 'Blues' or 'Reds') can look nice
    # to distinguish them from the points. Here we'll use a single color for simplicity.
    contour_color = 'navy'
    contour = ax.contour(xx, yy, Z, levels=10, colors=contour_color, linewidths=0.8)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')

    # Final styling touches
    ax.set_title("GMM Probability Density Contours", fontsize=16)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, linestyle=':', color='gray', alpha=0.6)
    ax.set_aspect('equal', adjustable='box') 

    plt.show()