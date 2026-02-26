import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from BI import bi
import jax.numpy as jnp
import jax
import jax.scipy.stats as stats
import numpyro

# Setup device------------------------------------------------
m = bi(platform='cpu')

# Generate Synthetic Data ------------------------------------
# 4 clusters
data, true_labels = make_blobs(
    n_samples=500, centers=4, cluster_std=0.8,
    center_box=(-10,10), random_state=101
)
N, D = data.shape

print(f"Data shape: {data.shape}")
print(f"True labels shape: {true_labels.shape}")


# BNN Mixture Model (Manual LogSumExp without SBP)-------------
print("\n--- Running BNN Mixture Model (Manual LogSumExp + Dirichlet) ---")
m.data_on_model = dict(X=data)

def unsupervised_mixture_model(X, D_H1=10, K=11):
    N, D_X = X.shape
    
    # 1. BNN: Learns to assign data to clusters (outputs mixing probabilities)
    w1 = m.bnn.layer_linear(
        X, 
        dist=m.dist.normal(0, 1, name='w1_weight', shape=(D_X, D_H1)),
        activation='tanh'
    )
    
    w2 = m.bnn.layer_linear(
        w1,
        dist=m.dist.normal(0, 0.05, name='w2_weight', shape=(D_H1, K))
    )
    
    # Avoid Stick-Breaking Process!
    # Instead, we use a single symmetric Dirichlet prior over the K mixture weights.
    # Alpha < 1 (e.g. 0.05) forces parsimony, acting exactly like a Dirichlet Process approximation
    alpha = 0.01
    pi = numpyro.sample('global_pi', numpyro.distributions.Dirichlet(jnp.ones(K) * (alpha / K)))
    
    # Add the logarithmic sparsity bounds to the BNN gating unit logits
    logits = w2 + jnp.log(pi + 1e-10)
    
    # 'p' is the soft cluster assignment for each observation (Shape: N x K)
    p = jnp.exp(jax.nn.log_softmax(logits, axis=-1))
    numpyro.deterministic('theta', p)
    
    # 2. Define the physical properties of the K clusters
    # Each cluster has a mean and standard deviation for the D_X features
    mu = m.dist.normal(0, 5, name='cluster_means', shape=(K, D_X))
    
    # User requested to test exponential prior as in their code block for cluster_stds
    sigma = m.dist.exponential(1.0, name='cluster_stds', shape=(K, D_X))
    
    # 3. The Likelihood: Marginalized Gaussian Mixture
    # --- MANUAL CALCULATION VIA LOGSUMEXP ---
    # a. Calculate log-probability of X under every cluster
    # Expand dims to evaluate all N observations against all K clusters
    # X shape: (N, D_X, 1) -> broadcasting requires (N, 1, D_X) compared to (1, K, D_X)
    X_exp = jnp.expand_dims(X, axis=1) # (N, 1, D_X)
    mu_exp = jnp.expand_dims(mu, axis=0) # (1, K, D_X)
    sigma_exp = jnp.expand_dims(sigma, axis=0) # (1, K, D_X)
    
    # log_pdf shape: (N, K, D_X). Sum across D_X -> log_pdf per observation per cluster: (N, K)
    log_pdf_clusters = jnp.sum(stats.norm.logpdf(X_exp, loc=mu_exp, scale=sigma_exp), axis=-1)
    
    # b. Weight the cluster log-probabilities by the BNN's predicted log-probabilities (log p)
    log_p = jnp.log(p + 1e-8)
    weighted_log_pdf = log_p + log_pdf_clusters
    
    # c. LogSumExp merges the K clusters back together (Marginalization)
    total_log_likelihood = jax.scipy.special.logsumexp(weighted_log_pdf, axis=-1)
    
    # Add to model target using numpyro factor
    numpyro.factor('mixture_likelihood', jnp.sum(total_log_likelihood))

m.fit(unsupervised_mixture_model, num_chains=1)


print("\n--- Extracting BNN Mixture Model Cluster Assignments ---")
theta_samps = m.posteriors['theta']
mu_samps = m.posteriors['cluster_means']
sigma_samps = m.posteriors['cluster_stds']

if theta_samps.ndim == 4:
    theta_samps = theta_samps.reshape(-1, *theta_samps.shape[2:])
    mu_samps = mu_samps.reshape(-1, *mu_samps.shape[2:])
    sigma_samps = sigma_samps.reshape(-1, *sigma_samps.shape[2:])

def get_bnn_cluster_probs_manual(data_pts, theta_prob, mu_p, sigma_p):
    # data_pts: (N, D_X)
    X_e = jnp.expand_dims(data_pts, axis=1) # (N, 1, D_X)
    mu_e = jnp.expand_dims(mu_p, axis=0) # (1, K, D_X)
    sig_e = jnp.expand_dims(sigma_p, axis=0) # (1, K, D_X)
    
    log_liks = jnp.sum(stats.norm.logpdf(X_e, loc=mu_e, scale=sig_e), axis=-1) # (N, K)
    log_probs = jnp.log(theta_prob + 1e-8) + log_liks
    norm_probs = jnp.exp(log_probs - jax.scipy.special.logsumexp(log_probs, axis=-1, keepdims=True))
    return norm_probs

cluster_probs = jax.vmap(get_bnn_cluster_probs_manual, in_axes=(None, 0, 0, 0))(data, theta_samps, mu_samps, sigma_samps)

# Consensus Clustering for BNN
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage

similarity_matrix = (cluster_probs @ cluster_probs.transpose(0, 2, 1)).mean(axis=0)
distance_matrix = 1 - similarity_matrix
distance_matrix = (distance_matrix + distance_matrix.T) / 2
distance_matrix = distance_matrix.at[jnp.diag_indices_from(distance_matrix)].set(0.0)
distance_matrix = jnp.clip(distance_matrix, a_min=0.0, a_max=None)

condensed_dist = squareform(distance_matrix)
Z = linkage(condensed_dist, 'average')
bnn_labels = fcluster(Z, t=0.5, criterion='distance')
print(f"BNN Mixture Model identified clusters: {len(np.unique(bnn_labels))}")

from sklearn.metrics import adjusted_rand_score
bnn_ari = adjusted_rand_score(true_labels, bnn_labels)
print(f"BNN Mixture Model Adjusted Rand Index: {bnn_ari:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(data[:, 0], data[:, 1], c=true_labels, cmap='viridis')
axes[0].set_title('True Labels')
axes[1].scatter(data[:, 0], data[:, 1], c=bnn_labels, cmap='viridis')
axes[1].set_title(f'BNN Manual LogSumExp (ARI: {bnn_ari:.3f})')
plt.savefig('cluster_comparison_manual.png')
plt.close()

print("\n--- Process Complete ---")

with open("ari_result.txt", "w") as f:
    f.write(f"Clusters: {len(np.unique(bnn_labels))}\n")
    f.write(f"ARI: {bnn_ari:.4f}\n")
