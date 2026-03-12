import jax
import sys
sys.path.insert(0, '/home/sosa/work/BI')

import jax.numpy as jnp
from sklearn.datasets import make_blobs
from BI import bi
from BI.ML.ClusteringEvaluator import ClusteringEvaluator

# 1. Generate the known clusters data as requested
data, true_labels = make_blobs(
    n_samples=500, 
    centers=4, 
    cluster_std=0.8,
    center_box=(-10, 10), 
    random_state=101
)
N, D = data.shape
print(f"Generated {N} samples with {D} features spread across 4 true centers.")

# Convert to JAX arrays
X = jnp.array(data)
labels_true = jnp.array(true_labels)

# 2. Instantiate BI
m = bi(platform='cpu')
evaluator = ClusteringEvaluator()

print("\n=======================================")
print("Testing DPMM with Empirical Bayes Priors")
print("=======================================")
sys.stdout.flush()
m.data_on_model = dict(data=data, T=10, empirical_bayes=True)
# Run MCMC
m.fit(m.models.dpmm, num_chains=1, num_samples=250, num_warmup=250, progress_bar=False)
print("Fit completed.")
sys.stdout.flush()

print("Running Consensus Predict to resolve label switching...")
# Predict labels
w_samps, mu_samps, sigma_samps, Lcorr_samps, pred_labels_dpmm = m.models.dpmm.predict(data, m.sampler)

print("\n--- DPMM Evaluation Metrics ---")
sil = evaluator.silhouette_score(X, pred_labels_dpmm)
db = evaluator.davies_bouldin_score(X, pred_labels_dpmm)
ch = evaluator.calinski_harabasz_score(X, pred_labels_dpmm)
ari = evaluator.adjusted_rand_score(labels_true, pred_labels_dpmm)

print(f"Silhouette: {sil:.4f}")
print(f"Davies-Bouldin: {db:.4f}")
print(f"Calinski-Harabasz: {ch:.4f}")
print(f"Adjusted Rand Index: {ari:.4f}")

print("\n=======================================")
print("Testing BNNC with Empirical Bayes Priors")
print("=======================================")
m.data_on_model = dict(data=data, K=11, D_H1=10, empirical_bayes=True)
m.fit(m.models.bnnc, progress_bar=False, num_chains=1, num_samples=250, num_warmup=250)

print("Running Consensus Predict to resolve label switching...")
theta_samps, mu_samps, sigma_samps, pred_labels_bnnc = m.models.bnnc.predict(data, m.sampler)

print("\n--- BNNC Evaluation Metrics ---")
sil_bnnc = evaluator.silhouette_score(X, pred_labels_bnnc)
db_bnnc = evaluator.davies_bouldin_score(X, pred_labels_bnnc)
ch_bnnc = evaluator.calinski_harabasz_score(X, pred_labels_bnnc)
ari_bnnc = evaluator.adjusted_rand_score(labels_true, pred_labels_bnnc)

print(f"Silhouette: {sil_bnnc:.4f}")
print(f"Davies-Bouldin: {db_bnnc:.4f}")
print(f"Calinski-Harabasz: {ch_bnnc:.4f}")
print(f"Adjusted Rand Index: {ari_bnnc:.4f}")
