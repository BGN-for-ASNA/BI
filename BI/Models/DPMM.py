from BI.Utils.np_dists import UnifiedDist as dist
import jax.numpy as jnp
import numpyro.distributions as Dist
import numpyro
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import multivariate_normal
import seaborn as sns # Ensure seaborn is imported for palettes
import matplotlib.pyplot as plt
import jax
import numpy as np

import numpyro

import jax.numpy as jnp

def mix_weights(beta):
    beta1m_cumprod = jnp.cumprod(1.0 - beta, axis=-1)
    padded_beta = jnp.pad(beta, (0, 1), constant_values=1.0)
    padded_cumprod = jnp.pad(beta1m_cumprod, (1, 0), constant_values=1.0)
    return padded_beta * padded_cumprod

def dpmm(data, T=10):
    N, D = data.shape  # Number of features
    data_mean = jnp.mean(data, axis=0)
    data_std = jnp.std(data, axis=0)*2

    # 1) stick-breaking weights
    alpha = dist.gamma(1.0, 10.0,name='alpha')

    with numpyro.plate("beta_plate", T - 1):
        beta = numpyro.sample('beta', Dist.Beta(1, alpha))

    w = numpyro.deterministic("w",mix_weights(beta))


    # 2) component parameters
    with numpyro.plate("components", T):
        mu = dist.multivariate_normal(loc=data_mean, covariance_matrix=data_std*jnp.eye(D),name='mu')# shape (T, D)        
        sigma = dist.log_normal(0.0, 1.0,shape=(D,),event=1,name='sigma')# shape (T, D)
        Lcorr = dist.lkj_cholesky(dimension=D, concentration=1.0,name='Lcorr')# shape (T, D, D)

        scale_tril = sigma[..., None] * Lcorr  # shape (T, D, D)

    # 3) Latent cluster assignments for each data point
    with numpyro.plate("data", N):
        # Sample the assignment for each data point
        z = numpyro.sample("z", Dist.Categorical(w)) # shape (N,)  

        numpyro.sample(
            "obs",
            Dist.MultivariateNormal(loc=mu[z], scale_tril=scale_tril[z]),
            obs=data
        )  

def dpmm_marginal(data, T=10):
    # 1) stick-breaking weights
    alpha = m.dist.gamma(1.0, 10.0,name='alpha')
    beta = m.dist.beta(1, alpha,name='beta',shape=(T-1,))
    w = numpyro.deterministic("w",dist.transforms.StickBreakingTransform()(beta))


    # 2) component parameters
    data_mean = jnp.mean(data, axis=0)
    with numpyro.plate("components", T):
        mu = m.dist.multivariatenormal(loc=data_mean, covariance_matrix=100.0*jnp.eye(D),name='mu')# shape (T, D)        
        sigma = m.dist.halfcauchy(1,shape=(D,),event=1,name='sigma')# shape (T, D)
        Lcorr = m.dist.lkjcholesky(dimension=D, concentration=1.0,name='Lcorr')# shape (T, D, D)

        scale_tril = sigma[..., None] * Lcorr  # shape (T, D, D)

    # 3) marginal mixture over obs
    m.dist.mixturesamefamily(
        mixing_distribution=m.dist.categoricalprobs(w,name='cat', create_obj=True),
        component_distribution=m.dist.multivariatenormal(loc=mu, scale_tril=scale_tril,name='mvn', create_obj=True),
        name="obs",  
        obs=data   
    )

def predict_dpmm(data, sampler):
    """
    Predicts the DPMM density contours based on posterior samples and final labels.
    Parameters:
    - data: The dataset used for prediction. Shape (N, D).
    - sampler: The sampler object containing posterior samples.
    Returns:
    - array of predicted labels for each data point.
    """

    # 1. Calculate posterior mean of all model parameters
    posterior_samples = sampler.get_samples()
    w_samps = posterior_samples['w']
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

    num_found_clusters = len(np.unique(final_labels))
    print(f"Model found {num_found_clusters} clusters.")

    return post_mean_w, post_mean_mu, post_mean_cov, final_labels

def plot_dpmm(data,sampler,figsize=(10, 8), point_size=10):
    post_mean_w, post_mean_mu, post_mean_cov, final_labels = predict_dpmm(data,sampler)
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
    fig, ax = plt.subplots(figsize=figsize)
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
    ax.scatter(data[:, 0], data[:, 1], c=point_colors, s=point_size, alpha=0.9, edgecolor='white', linewidth=0.3)

    # Plot the density contours
    # Using a different colormap for the contours (e.g., 'Blues' or 'Reds') can look nice
    # to distinguish them from the points. Here we'll use a single color for simplicity.
    contour_color = 'navy'
    contour = ax.contour(xx, yy, Z, levels=10, colors=contour_color, linewidths=0.8)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')

    # Final styling touches
    ax.set_title("DPMM Probability Density Contours", fontsize=16)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, linestyle=':', color='gray', alpha=0.6)
    #ax.set_aspect('equal', adjustable='box') 

    plt.show()
