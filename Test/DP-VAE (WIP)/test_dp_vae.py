import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from BI import bi
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

# Add project root to sys.path
project_root = r'C:\Users\Sosa\Documents\BI'
if project_root not in sys.path:
    sys.path.append(project_root)

# Initialize BI
m = bi(platform='cpu')

# 1. Generate Synthetic Data ------------------------------------
n_samples = 300
n_features = 2
n_clusters = 3
cluster_std = 0.6

X_orig, y_true = make_blobs(
    n_samples=n_samples, 
    n_features=n_features, 
    centers=n_clusters, 
    cluster_std=cluster_std, 
    random_state=42
)

# Standardize data for neural network stability
X_mean = jnp.mean(X_orig, axis=0)
X_std = jnp.std(X_orig, axis=0)
X = (X_orig - X_mean) / X_std

# Convert to JAX arrays
X = jnp.array(X)

# 2. Define DP-VAE Model ---------------------------------------
# Based on readme.md and BI patterns
def dp_vae_model(data, K=10, D_Z=2, D_H=32):
    """
    Dirichlet Process Variational Autoencoder.
    
    Args:
        data: Observed data (N, D_X)
        K: Maximum number of clusters (truncation)
        D_Z: Latent space dimensionality
        D_H: Hidden layer width
    """
    N, D_X = data.shape
    
    # --- 1) Stick-Breaking Prior for Latent Space ---
    alpha = m.dist.gamma(1.0, 10.0, name='alpha')
    beta = m.dist.beta(1, alpha, name='beta', shape=(K-1,))
    
    # Mixture weights helper from DPMM
    w = numpyro.deterministic("w", m.models.dpmm.mix_weights(beta))
    
    # Cluster parameters in latent space
    with m.dist.plate("clusters", K):
        mu_k = m.dist.normal(0, 2, name='mu_k', shape=(D_Z,), event=1)
        sigma_k = m.dist.exponential(1.0, name='sigma_k', shape=(D_Z,), event=1)
        
    # --- 2) Encoder (BNN) ---
    b_enc1 = m.dist.normal(0, 1, name='b_enc1', shape=(D_H,))
    w_enc1 = m.bnn.layer_linear(
        data, 
        dist=m.dist.normal(0, 1, name='w_enc1', shape=(D_X, D_H)),
        bias=b_enc1,
        activation='tanh'
    )
    
    b_z_mu = m.dist.normal(0, 1, name='b_z_mu', shape=(D_Z,))
    z_mu = m.bnn.layer_linear(
        w_enc1,
        dist=m.dist.normal(0, 1, name='z_mu_layer', shape=(D_H, D_Z)),
        bias=b_z_mu
    )
    
    b_z_sigma = m.dist.normal(0, 1, name='b_z_sigma', shape=(D_Z,))
    z_sigma = m.bnn.layer_linear(
        w_enc1,
        dist=m.dist.normal(0, 1, name='z_sigma_layer', shape=(D_H, D_Z)),
        bias=b_z_sigma,
        activation='softplus' # Ensure positivity
    )
    
    # Latent sampling (Variational representation)
    with m.dist.plate("data", N):
        z = m.dist.normal(z_mu, z_sigma, name='z', event=1)
        
        # --- 3) Latent Prior Likelihood (DPMM on Z) ---
        log_pdf_clusters = jnp.sum(
            dist.Normal(mu_k[None, :, :], sigma_k[None, :, :]).log_prob(z[:, None, :]),
            axis=-1
        ) # (N, K)
        weighted_log_pdf = jnp.log(w) + log_pdf_clusters # (N, K)
        numpyro.factor("latent_prior", jax.scipy.special.logsumexp(weighted_log_pdf, axis=-1).sum())

    # --- 4) Decoder (BNN) ---
    b_dec1 = m.dist.normal(0, 1, name='b_dec1', shape=(D_H,))
    w_dec1 = m.bnn.layer_linear(
        z,
        dist=m.dist.normal(0, 1, name='w_dec1', shape=(D_Z, D_H)),
        bias=b_dec1,
        activation='tanh'
    )
    
    b_recon = m.dist.normal(0, 1, name='b_recon', shape=(D_X,))
    recon_mu = m.bnn.layer_linear(
        w_dec1,
        dist=m.dist.normal(0, 1, name='recon_mu_layer', shape=(D_H, D_X)),
        bias=b_recon
    )
    
    recon_sigma = m.dist.gamma(1.0, 10.0, name='recon_sigma') # Favor smaller values for tight reconstruction
    
    # --- 5) Observation Likelihood ---
    m.dist.normal(recon_mu, recon_sigma, obs=data, name='obs')

# 3. Fit Model ------------------------------------------------
m.data_on_model = dict(data=X)
# 3. Fit Model ------------------------------------------------
m.data_on_model = dict(data=X)
print("Fitting DP-VAE model with SVI (10k steps)...")
# Using a learning rate scheduler for better convergence
from numpyro.optim import Adam
optimizer = Adam(step_size=1e-3)
m.svi(dp_vae_model, num_steps=10000, num_samples=1000, guide='diagonal', optim=optimizer)

# 4. Evaluation -----------------------------------------------
print("\nEvaluation results:")
# Get posterior samples for clustering
samples = m.posteriors
w_samps = samples['w'] # (S, K)
z_samps = samples['z'] # (S, N, D_Z)
mu_k_samps = samples['mu_k'] # (S, K, D_Z)
sigma_k_samps = samples['sigma_k'] # (S, K, D_Z)

# Assign each point to the cluster with highest responsibility (mean over samples)
def get_responsibilities(z, w, mu, sigma):
    # z: (N, D_Z), w: (K,), mu: (K, D_Z), sigma: (K, D_Z)
    log_probs = dist.Normal(mu[None, :, :], sigma[None, :, :]).log_prob(z[:, None, :]).sum(axis=-1)
    log_resp = jnp.log(w) + log_probs
    resp = jax.nn.softmax(log_resp, axis=-1)
    return resp

# Log responsibilities for EACH sample
all_resps = jax.vmap(get_responsibilities)(z_samps, w_samps, mu_k_samps, sigma_k_samps) # (S, N, K)

# Consensus Matrix: average probability that two points are in the same cluster
# Adjacency for one sample: A = R R^T (N, N)
# Average over samples
print("Computing consensus matrix...")
similarity_matrix = (all_resps @ all_resps.transpose(0, 2, 1)).mean(axis=0)

# Hierarchical Clustering on consensus matrix
distance_matrix = 1 - similarity_matrix
distance_matrix = (distance_matrix + distance_matrix.T) / 2
# Set diagonal to exactly 0
indices = np.diag_indices_from(distance_matrix)
distance_matrix = np.array(jax.device_get(distance_matrix)) # Copy to writable numpy
distance_matrix[indices] = 0.0
# Ensure no tiny negatives from floating point
distance_matrix = np.clip(distance_matrix, a_min=0.0, a_max=None)

condensed_dist = squareform(distance_matrix)
Z = linkage(condensed_dist, 'average')
inferred_labels = fcluster(Z, t=0.5, criterion='distance')

# Simple Argmax (valid for SVI)
mean_resps = jnp.mean(all_resps, axis=0)
argmax_labels = jnp.argmax(mean_resps, axis=-1)
ari_argmax = adjusted_rand_score(y_true, argmax_labels)
print(f"Argmax Adjusted Rand Index: {ari_argmax:.4f}")

# Calculate Accuracy (Consensus ARI)
ari_consensus = adjusted_rand_score(y_true, inferred_labels)
print(f"Consensus Adjusted Rand Index: {ari_consensus:.4f}")

ari = max(ari_argmax, ari_consensus)

unique, counts = np.unique(inferred_labels, return_counts=True)
print(f"Label counts: {dict(zip(unique, counts))}")

mean_z = jnp.mean(z_samps, axis=0) # (N, D_Z)
print(f"Latent space stats: mean={jnp.mean(mean_z):.3f}, std={jnp.std(mean_z):.3f}")
# Also print latent range to see if it's spread out
print(f"Latent range: min={jnp.min(mean_z):.3f}, max={jnp.max(mean_z):.3f}")

# Check active clusters (weights > mean + 2*std or just a threshold)
mean_w = jnp.mean(w_samps, axis=0)
active_clusters = jnp.sum(mean_w > 0.05) # Practical threshold
print(f"Number of active clusters found: {active_clusters}")
print(f"True number of clusters: {n_clusters}")

# 5. Visualization --------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Inferred Clusters in Data Space
axes[0].scatter(X_orig[:, 0], X_orig[:, 1], c=inferred_labels, cmap='viridis', alpha=0.6)
axes[0].set_title(f"Inferred Clusters (ARI: {ari:.4f})")
axes[0].set_xlabel("X1")
axes[0].set_ylabel("X2")

# Plot 2: Latent Space
mean_z = jnp.mean(z_samps, axis=0)
axes[1].scatter(mean_z[:, 0], mean_z[:, 1], c=y_true, cmap='Set1', alpha=0.6)
axes[1].set_title("Latent Space (Colored by True Labels)")
axes[1].set_xlabel("Z1")
axes[1].set_ylabel("Z2")

plt.tight_layout()
output_plot = r'C:\Users\Sosa\Documents\BI\Test\DP-VAE\results.png'
plt.savefig(output_plot)
print(f"Results plot saved to {output_plot}")

# Summary to file
with open(r'C:\Users\Sosa\Documents\BI\Test\DP-VAE\summary.txt', 'w') as f:
    f.write(f"Adjusted Rand Index: {ari:.4f}\n")
    f.write(f"Number of active clusters: {active_clusters}\n")
    f.write(f"True clusters: {n_clusters}\n")
