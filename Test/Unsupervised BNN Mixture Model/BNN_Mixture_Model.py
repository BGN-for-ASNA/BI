import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from BI import bi
import jax.numpy as jnp
import jax
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

plt.scatter(data[:, 0], data[:, 1], c=true_labels, cmap='viridis')
plt.title('Synthetic Data (True Labels)')
plt.savefig('synthetic_data.png')
plt.close()

# DPMM Baseline-----------------------------------------------
print("\n--- Running DPMM Baseline ---")
m.data_on_model = dict(data=data, T=11)
m.fit(m.models.dpmm, num_chains=1)

dpmm_clusters = m.models.dpmm.proportion_of_data_assigned_to_cluster()

print("\n--- Extracting DPMM Cluster Assignments ---")
w_samps_dpmm, mu_samps_dpmm, sigma_samps_dpmm, Lcorr_samps_dpmm, dpmm_labels = m.models.dpmm.predict(data, m.sampler)
from sklearn.metrics import adjusted_rand_score
dpmm_ari = adjusted_rand_score(true_labels, dpmm_labels)
print(f"DPMM Adjusted Rand Index: {dpmm_ari:.4f}")

# BNN Mixture Model---------------------------------------------
print("\n--- Running BNN Mixture Model ---")
m.data_on_model = dict(data=data)

def bnn_mixture_model(data, K=11, D_H1=10):
    # data shape: (N, D)
    N, D_X = data.shape
    
    # --- BNN Gating Network ---
    # First hidden layer
    w1 = m.bnn.layer_linear(
        data, 
        dist=m.dist.normal(0, 1, name='w1_weight', shape=(D_X, D_H1)),
        activation='tanh'
    )
    
    # Output layer -> Logits for K classes
    # Output layer -> Logits for K classes
    w2 = m.bnn.layer_linear(
        w1,
        dist=m.dist.normal(0, 0.05, name='w2_weight', shape=(D_H1, K))
    )
    
    # Global parsimony prior: Stick-Breaking Process (SBP) like DPMM
    # This explicitly orders global cluster sizes, favoring the first few and aggressively shrinking the rest.
    alpha = 0.05 # DP concentration parameter
    v = numpyro.sample('v', numpyro.distributions.Beta(1.0, alpha).expand([K-1]))
    
    # Construct stick breaking proportions pi
    def stick_breaking(v):
        cum_v = jnp.cumprod(1 - v)
        v_one = jnp.pad(v, (0, 1), constant_values=1.0)
        cum_v_pad = jnp.pad(cum_v, (1, 0), constant_values=1.0)
        return v_one * cum_v_pad
        
    pi = stick_breaking(v)
    numpyro.deterministic('global_pi', pi)
    
    # Softmax to get mixing probabilities (theta)
    # Adding log(global_pi) to the logits acts as a sparse bias to suppress inactive clusters
    logits = w2 + jnp.log(pi + 1e-10)
    theta = jnp.exp(jax.nn.log_softmax(logits, axis=-1))
    numpyro.deterministic('theta', theta)
    
    # --- Mixture Components ---
    # Component means (K, D)
    mu = m.dist.normal(0, 5, name='mu', shape=(K, D_X))
    
    # Component scales (K, D)
    sigma = m.dist.half_normal(1, name='sigma', shape=(K, D_X))
    
    # Likelihood: The data is a mixture of K multivariate normals,
    # mixed according to the BNN outputs `theta`.
    m.dist.mixture_same_family(
        mixing_distribution=numpyro.distributions.Categorical(probs=theta),
        component_distribution=numpyro.distributions.Independent(
            numpyro.distributions.Normal(mu, sigma), reinterpreted_batch_ndims=1
        ),
        obs=data
    )

m.fit(bnn_mixture_model, num_chains=1)

print("\n--- Extracting BNN Mixture Model Cluster Assignments ---")
theta_samps = m.posteriors['theta']
mu_samps = m.posteriors['mu']
sigma_samps = m.posteriors['sigma']

if theta_samps.ndim == 4: # Handle multiple chains if needed
    theta_samps = theta_samps.reshape(-1, *theta_samps.shape[2:])
    mu_samps = mu_samps.reshape(-1, *mu_samps.shape[2:])
    sigma_samps = sigma_samps.reshape(-1, *sigma_samps.shape[2:])

def get_bnn_cluster_probs(data_pts, theta_prob, mu_p, sigma_p):
    log_liks = numpyro.distributions.Independent(
        numpyro.distributions.Normal(mu_p, sigma_p), 1
    ).log_prob(data_pts[:, None, :])
    log_probs = jnp.log(theta_prob) + log_liks
    norm_probs = jnp.exp(log_probs - jax.scipy.special.logsumexp(log_probs, axis=-1, keepdims=True))
    return norm_probs

cluster_probs = jax.vmap(get_bnn_cluster_probs, in_axes=(None, 0, 0, 0))(data, theta_samps, mu_samps, sigma_samps)

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

bnn_ari = adjusted_rand_score(true_labels, bnn_labels)
print(f"BNN Mixture Model Adjusted Rand Index: {bnn_ari:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].scatter(data[:, 0], data[:, 1], c=true_labels, cmap='viridis')
axes[0].set_title('True Labels')
axes[1].scatter(data[:, 0], data[:, 1], c=dpmm_labels, cmap='viridis')
axes[1].set_title(f'DPMM (ARI: {dpmm_ari:.3f})')
axes[2].scatter(data[:, 0], data[:, 1], c=bnn_labels, cmap='viridis')
axes[2].set_title(f'BNN (ARI: {bnn_ari:.3f})')
plt.savefig('cluster_comparison.png')
plt.close()

print("\n--- Process Complete ---")
