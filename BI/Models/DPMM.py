from BI.Distributions.np_dists import UnifiedDist as dist
import jax.numpy as jnp
import numpyro.distributions as Dist
import numpyro
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
import jax

import numpyro

import jax.numpy as jnp

dist = dist()

def mix_weights(beta):
    """
    Mixture weights (stick-breaking) for DPMM.
    The stick-breaking weights are used to sample the mixture component assignments.
    Args:
        beta: Stick-breaking weights (T-1 components)
    Returns:
        w: Mixture weights (T components)
    Reference:
        https://pyro.ai/examples/dirichlet_process_mixture.html
    """
    beta1m_cumprod = jnp.cumprod(1.0 - beta, axis=-1)
    padded_beta = jnp.pad(beta, (0, 1), constant_values=1.0)
    padded_cumprod = jnp.pad(beta1m_cumprod, (1, 0), constant_values=1.0)
    return padded_beta * padded_cumprod

def dpmm_latent(data, T=10):
    """
    Latent Variable formulation: Explicitly samples 'z'.
    Requires a sampler that supports discrete variables (e.g., MixedHMC or DiscreteHMCGibbs).
    """
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
    """
    Marginalized formulation: Integrates out 'z'.
    Standard formulation for NUTS/HMC samplers.
    """

    D = data.shape[1]
    # 1) stick-breaking weights
    alpha = dist.gamma(1.0, 15.0,name='alpha')
    beta = dist.beta(1, alpha,name='beta',shape=(T-1,))
    w = numpyro.deterministic("w",mix_weights(beta))


    # 2) component parameters
    data_mean = jnp.mean(data, axis=0)
    with numpyro.plate("components", T):
        mu = dist.multivariate_normal(loc=data_mean, covariance_matrix=100.0*jnp.eye(D),name='mu')# shape (T, D)        
        sigma = dist.half_cauchy(1,shape=(D,),event=1,name='sigma')# shape (T, D)
        Lcorr = dist.lkj_cholesky(dimension=D, concentration=1.0,name='Lcorr')# shape (T, D, D)

        scale_tril = sigma[..., None] * Lcorr  # shape (T, D, D)

    # 3) marginal mixture over obs
    dist.mixture_same_family(
        mixing_distribution=dist.categorical_probs(w,name='cat', create_obj=True),
        component_distribution=dist.multivariate_normal(loc=mu, scale_tril=scale_tril,name='mvn', create_obj=True),
        name="obs",  
        obs=data   
    )


def dpmm(data, T=10, method='marginal'):
    """
    Wrapper function for DPMM.
    
    Args:
        data: Input data array (N, D)
        T: Truncation level (max number of clusters)
        method: 'marginal' (default, for NUTS) or 'latent' (explicit z, requires MixedHMC/Gibbs)
    """
    if method == 'marginal':
        return dpmm_marginal(data, T)
    elif method == 'latent':
        return dpmm_latent(data, T)
    else:
        raise ValueError("Method must be 'marginal' or 'latent'")
    
def predict_dpmm(data, sampler):
    """
    Predicts the DPMM density contours based on posterior samples and final labels.
    Parameters:
    - data: The dataset used for prediction. Shape (N, D).
    - sampler: The sampler object containing posterior samples.
    Returns:
    - array of predicted labels for each data point.
    Details:

    This approach implements **Posterior Similarity Clustering** (also known as Consensus Clustering) to resolve the **Label Switching Problem** inherent in Bayesian Mixture Models.

    In Bayesian mixture models (like DPMM), the identity of a cluster is not fixed across MCMC samples (e.g., "Cluster 1" in sample $t$ might swap parameters with "Cluster 2" in sample $t+1$). Therefore, you cannot simply average the cluster assignments or parameters directly.

    The code solves this using the following pipeline:

    1.  **Soft Assignment Calculation:** For every MCMC sample and every data point, it calculates the probability of the point belonging to each mixture component (cluster).
    2.  **Co-occurrence Matrix Construction:** It computes an $N \times N$ pairwise similarity matrix. Each entry $(i, j)$ represents the marginal posterior probability that data point $i$ and    data point $j$ belong to the **same** cluster, averaged over all MCMC samples. This metric is invariant to label switching (it doesn't matter if they are both in cluster 1 or both in cluster     5, as long as they are together).
    3.  **Transformation to Distance:** The similarity matrix is inverted to create a distance matrix ($Distance = 1 - Probability$). Points that are rarely in the same cluster become "far apart."
    4.  **Hierarchical Clustering:** It applies standard Agglomerative Hierarchical Clustering (Average Linkage) on this distance matrix to group the data.
    5.  **Final Partitioning:** The hierarchical tree is cut at a specific threshold (0.5) to produce the final, hard cluster assignments for the dataset.

    ### References
    Medvedovic, M., & Sivaganesan, S. (2002). "Bayesian partition for clustering of expression data." Bioinformatics, 18(suppl_1), S74-S81.
    Dahl, D. B. (2006). "Model-Based Clustering for Expression Data via a Dirichlet Process Mixture Model." Bayesian Inference for Gene Expression and Proteomics, 201-218.
    """
    print("⚠️This function is still in development. Use it with caution. ⚠️")

    # --- 2. Define Helper Function for Soft Assignments ---
    posterior_samples = sampler.get_samples()

    w_samps = posterior_samples['w']          # Mixture weights (N_samples x N_components)
    mu_samps = posterior_samples['mu']        # Cluster means (N_samples x N_components x N_dims)
    Lcorr_samps = posterior_samples['Lcorr']  # Cholesky correlation matrices
    sigma_samps = posterior_samples['sigma']  # Scale (variance) parameters

    # --- 2. Define Helper Function for Soft Assignments ---
    def get_cluster_probs(data, w, mu, sigma, Lcorr):
        # Construct the lower triangular Cholesky factor of the covariance matrix
        # Combines standard deviations (sigma) with the correlation structure (Lcorr)
        scale_tril = sigma[..., None] * Lcorr

        # Compute the Log-Likelihood of the data under a Multivariate Normal distribution
        # Shape becomes (N_data, N_components)
        log_liks = Dist.MultivariateNormal(mu, scale_tril=scale_tril).log_prob(data[:, None, :])
        # Calculate unnormalized log posterior probabilities: log(weight) + log(Likelihood)
        log_probs = jnp.log(w) + log_liks

        # Normalize probabilities (Softmax) using LogSumExp trick for numerical stability
        # Result is P(z=k | data, parameters)
        norm_probs = jnp.exp(log_probs - jax.scipy.special.logsumexp(log_probs, axis=-1, keepdims=True))
        return norm_probs
    
    # --- 3. Compute Posterior Similarity Matrix (Consensus Matrix) ---
    # Use JAX vmap to vectorize the `get_cluster_probs` function.
    # This applies the function to all MCMC samples simultaneously without a Python loop.
    # Result shape: (N_samples, N_data_points, N_components)
    cluster_probs = jax.vmap(get_cluster_probs, in_axes=(None, 0, 0, 0, 0))(
        data, w_samps, mu_samps, sigma_samps, Lcorr_samps
    )

    

    # Calculate the Co-occurrence (Similarity) Matrix.
    # 1. cluster_probs @ cluster_probs.T computes the probability that point i and point j 
    #    are in the same cluster for a specific sample.
    # 2. .mean(axis=0) averages this over all MCMC samples.
    # Result is an (N_data x N_data) matrix where value (i,j) is the marginal probability
    # that points i and j belong to the same cluster.
    similarity_matrix = (cluster_probs @ cluster_probs.transpose(0, 2, 1)).mean(axis=0)

    # --- 5. Hierarchical Clustering on Similarity ---
    # Convert similarity (0 to 1) into a distance metric (0 means identical, 1 means distinct)
    distance_matrix = 1 - similarity_matrix

    # Symmetrize the matrix to ensure d(i,j) == d(j,i) (corrects floating point errors)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    # Enforce that the distance from a point to itself is exactly 0.0
    distance_matrix = distance_matrix.at[jnp.diag_indices_from(distance_matrix)].set(0.0)  # Set diagonal to 0

    # Clip values to ensure no negative distances due to floating point math
    distance_matrix = jnp.clip(distance_matrix, a_min=0.0, a_max=None)

    # Convert the square distance matrix to a condensed vector form required by scipy.linkage
    condensed_dist = squareform(distance_matrix)

    # Perform Agglomerative (Hierarchical) Clustering using 'average' linkage.
    # This builds a dendrogram based on how often points appeared in the same cluster.
    Z = linkage(condensed_dist, 'average')

    # --- 6. Determine Final Labels ---
    # Define a threshold to cut the dendrogram. 
    # 0.5 implies we group points if they appear together >50% of the time.
    distance_threshold = 0.5 

    # Extract flat cluster labels by cutting the tree at the threshold
    final_labels = fcluster(Z, t=distance_threshold, criterion='distance')

    num_found_clusters = len(np.unique(final_labels))
    print(f"Model found {num_found_clusters} clusters.")

    # --- 7. Posterior Means for Plotting ---
    post_mean_w = jnp.mean(w_samps, axis=0)
    post_mean_mu =jnp.mean(mu_samps, axis=0)
    post_mean_sigma = jnp.mean(sigma_samps, axis=0)
    post_mean_Lcorr = jnp.mean(Lcorr_samps, axis=0)

    # Reconstruct the full covariance matrices
    post_mean_scale_tril = post_mean_sigma[..., None] * post_mean_Lcorr
    post_mean_cov = post_mean_scale_tril @ jnp.transpose(post_mean_scale_tril, (0, 2, 1))

    return post_mean_w, post_mean_mu, post_mean_cov, final_labels
import matplotlib.pyplot as plt
import numpy as np


def proportion_of_data_assigned_to_cluster(cluster_probs):
    # 1. Process Data
    # Input shape: (N_samples, N_data, N_max_groups)
    # We take the mean over axis 1 (N_data) to get the proportion of the dataset 
    # assigned to each cluster for every MCMC sample.
    # Result shape: (N_samples, N_max_groups)
    cluster_proportions = np.array(cluster_probs).mean(axis=1)
    
    N_max_groups = cluster_proportions.shape[1]
    indices = np.arange(N_max_groups)
    
    # 2. Create the Plot
    plt.figure(figsize=(15, 6))
    
    # Create a boxplot. 
    # We pass the array transpose so that each column (Cluster) is treated as a dataset.
    plt.boxplot(cluster_proportions, positions=indices, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=1.5),
                flierprops=dict(marker='o', markersize=2, alpha=0.5))
    
    # 3. Formatting
    plt.xlabel("Cluster Index (Component ID)", fontsize=12)
    plt.ylabel("Proportion of Data Assigned (Posterior Mass)", fontsize=12)
    plt.title("Posterior Distribution of Cluster Sizes across MCMC Samples", fontsize=14)
    
    # Set X-ticks to match the cluster indices
    plt.xticks(indices, indices)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05) # Proportions are between 0 and 1
    
    plt.show()

def plot_dpmm(data,sampler,figsize=(10, 8), point_size=10):
    print("⚠️This function is still in development. Use it with caution. ⚠️")
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
